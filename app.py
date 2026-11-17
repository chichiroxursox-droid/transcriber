"""Transcriber — local web app. FastAPI + one WebSocket + one static page.

Threads (mic callback, audiotee readers, engine) push events through a single
thread-safe door: emit() → EVENTS_Q → broadcaster() (the only consumer, so no
locks) → WebSocket clients + server-side session state. flush_loop persists
finals to transcript.md + Google Doc; question_loop generates live questions.
"""
from __future__ import annotations

import os
import re
import json
import queue
import asyncio
import threading
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

import engine
import gdoc
import questions

load_dotenv()

HERE = Path(__file__).parent
TRANSCRIPTS = HERE / "transcripts"
PORT = int(os.environ.get("PORT", 8737))

LOOP: asyncio.AbstractEventLoop | None = None
EVENTS_Q: asyncio.Queue | None = None
CLIENTS: set[WebSocket] = set()
SESSION: "Session | None" = None
_LIFECYCLE = asyncio.Lock()          # serializes start/stop so they never interleave


@dataclass
class Session:
    title: str
    dir: Path
    engine: str
    provider: str
    use_mic: bool
    use_system: bool
    started_at: float
    stop_flag: threading.Event = field(default_factory=threading.Event)
    sources: list = field(default_factory=list)          # [(src, queue, wav_path)]
    srcs: list = field(default_factory=list)              # MicSource/SystemSource objects
    engine_started: bool = False
    engine_done: threading.Event = field(default_factory=threading.Event)
    finals: list = field(default_factory=list)           # [{source,text,t}]
    questions: list = field(default_factory=list)
    pending: list = field(default_factory=list)           # finals not yet flushed
    doc_id: str | None = None
    doc_url: str | None = None
    doc_fail: int = 0
    tasks: list = field(default_factory=list)

    @property
    def md_path(self) -> Path:
        return self.dir / "transcript.md"


def emit(ev: dict):
    """The only cross-thread door into the event loop."""
    if LOOP and EVENTS_Q:
        LOOP.call_soon_threadsafe(EVENTS_Q.put_nowait, ev)


def _state_ev() -> dict:
    s = SESSION
    return {"type": "state",
            "running": bool(s and s.engine_started and not s.engine_done.is_set()),
            "engine": s.engine if s else os.environ.get("ENGINE", "parakeet"),
            "provider": s.provider if s else os.environ.get("QUESTION_PROVIDER", "anthropic"),
            "title": s.title if s else "",
            "started_at": s.started_at if s else None,
            "doc_url": s.doc_url if s else None,
            "dir": str(s.dir) if s else None}


async def _broadcast(ev: dict):
    dead = []
    for ws in list(CLIENTS):          # snapshot: connect/disconnect may mutate CLIENTS mid-await
        try:
            await ws.send_json(ev)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


async def broadcaster():
    """Single consumer of EVENTS_Q: mutate session state, fan out to clients.
    Must never die — it's the sole consumer, so a crash would freeze all output."""
    while True:
        ev = await EVENTS_Q.get()
        try:
            s = SESSION
            if s:
                if ev["type"] == "final":
                    s.finals.append(ev)
                    s.pending.append(ev)
                elif ev["type"] == "questions":
                    s.questions = ev["items"]
            await _broadcast(ev)
        except Exception:
            pass


def _fmt_final(ev: dict) -> str:
    who = "You" if ev["source"] == "you" else "Them"
    return f"**[{ev['t']}] {who}:** {ev['text']}"


async def flush_loop(s: Session):
    """Every ~10s append newly-committed finals to transcript.md + Google Doc."""
    while True:
        await asyncio.sleep(10)
        if not s.pending:
            continue
        batch, s.pending = s.pending, []
        text = "\n\n".join(_fmt_final(e) for e in batch) + "\n\n"
        try:
            with open(s.md_path, "a") as f:
                f.write(text)
        except Exception as e:
            s.pending[:0] = batch          # re-queue so nothing is lost; retry next tick
            emit({"type": "error", "msg": f"transcript write failed: {str(e)[:120]}"})
            continue
        if s.doc_id and s.doc_fail < 3:
            try:
                await gdoc.append(s.doc_id, text)
                s.doc_fail = 0
            except Exception as e:
                s.doc_fail += 1
                if s.doc_fail >= 3:
                    emit({"type": "error", "msg": f"Google Doc sync disabled after errors: {e}"})


async def question_loop(s: Session):
    """Every ~20s, if new speech arrived, refresh suggested questions."""
    last_n = 0
    while True:
        await asyncio.sleep(20)
        if len(s.finals) <= last_n:
            continue
        last_n = len(s.finals)
        tail = "\n".join(f"{'You' if e['source']=='you' else 'Them'}: {e['text']}"
                         for e in s.finals[-40:])
        try:
            items, ms = await questions.generate(tail, s.questions, s.provider)
            if items:
                emit({"type": "questions", "items": items, "provider": s.provider, "ms": ms})
        except Exception as e:
            emit({"type": "error", "msg": f"question gen ({s.provider}): {str(e)[:160]}"})


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "session"


async def start_session(cfg: dict):
    global SESSION
    async with _LIFECYCLE:
        if SESSION and SESSION.engine_started and not SESSION.engine_done.is_set():
            emit({"type": "error", "msg": "a session is already running"}); return
        title = cfg.get("title") or "Untitled"
        now = datetime.now()
        d = TRANSCRIPTS / f"{now:%Y-%m-%d_%H%M}_{_slug(title)}"
        d.mkdir(parents=True, exist_ok=True)
        s = Session(title=title, dir=d,
                    engine=cfg.get("engine", os.environ.get("ENGINE", "parakeet")),
                    provider=os.environ.get("QUESTION_PROVIDER", "anthropic"),
                    use_mic=cfg.get("mic", True), use_system=cfg.get("system", True),
                    started_at=now.timestamp())
        SESSION = s
        try:
            with open(s.md_path, "w") as f:
                f.write(f"# {title}\n\n_{now:%Y-%m-%d %H:%M}_\n\n")

            # Google Doc (best-effort; transcript.md is the source of truth)
            try:
                s.doc_id, s.doc_url = await gdoc.create_doc(f"Transcript — {title} ({now:%Y-%m-%d %H:%M})")
                await gdoc.append(s.doc_id, f"Transcript — {title}\n{now:%Y-%m-%d %H:%M}\n\n")
            except Exception as e:
                emit({"type": "error", "msg": f"Google Doc unavailable: {str(e)[:160]}"})

            from audio import MicSource, SystemSource
            if s.use_mic:
                q = queue.Queue(); s.sources.append(("you", q, str(d / "you.wav")))
                mic = MicSource(q, emit); mic.start(); s.srcs.append(mic)
            if s.use_system:
                q = queue.Queue(); s.sources.append(("them", q, str(d / "them.wav")))
                sysrc = SystemSource(q, emit); sysrc.start(); s.srcs.append(sysrc)

            s.engine_started = True
            engine.submit(s, emit)          # runs on the single persistent engine thread
            s.tasks = [asyncio.create_task(flush_loop(s)), asyncio.create_task(question_loop(s))]
        except Exception as e:
            for src in s.srcs:                 # unwind any audio capture we opened
                try: src.stop()
                except Exception: pass
            s.stop_flag.set()
            SESSION = None
            emit({"type": "error", "msg": f"failed to start session: {str(e)[:160]}"})
            await _broadcast(_state_ev())
            return
        await _broadcast(_state_ev())


async def stop_session():
    async with _LIFECYCLE:
        s = SESSION
        if not s:
            return
        for src in s.srcs:
            src.stop()
        s.stop_flag.set()
        if s.engine_started:
            await asyncio.to_thread(s.engine_done.wait, 15)
        for t in s.tasks:
            t.cancel()
        # final flush of anything pending
        if s.pending:
            text = "\n\n".join(_fmt_final(e) for e in s.pending) + "\n\n"
            try:
                with open(s.md_path, "a") as f:
                    f.write(text)
                if s.doc_id and s.doc_fail < 3:
                    await gdoc.append(s.doc_id, text)
            except Exception:
                pass
            s.pending = []
        (s.dir / "meta.json").write_text(json.dumps({
            "title": s.title, "engine": s.engine, "provider": s.provider,
            "started_at": s.started_at, "ended_at": datetime.now().timestamp(),
            "doc_url": s.doc_url, "finals": len(s.finals),
            "sources": [src for src, _q, _w in s.sources]}, indent=2))
        await _broadcast(_state_ev())


async def summarize():
    s = SESSION
    if not s or not s.finals:
        emit({"type": "error", "msg": "nothing to summarize yet"}); return
    tail = "\n".join(f"{'You' if e['source']=='you' else 'Them'}: {e['text']}" for e in s.finals)
    prompt = ("Summarize this conversation transcript in a few tight bullet points, then a "
              "one-line list of any action items or open questions.\n\n" + tail[-12000:])
    try:
        from questions import _via_api, _via_cli
        raw = await (_via_cli(prompt) if s.provider == "claude_cli" else _via_api(prompt))
    except Exception as e:
        emit({"type": "error", "msg": f"summary failed: {str(e)[:160]}"}); return
    block = f"\n\n## Summary\n\n{raw.strip()}\n"
    with open(s.md_path, "a") as f:
        f.write(block)
    if s.doc_id:
        try:
            await gdoc.append(s.doc_id, block)
        except Exception:
            pass
    emit({"type": "summary", "text": raw.strip()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOOP, EVENTS_Q
    LOOP = asyncio.get_running_loop()
    EVENTS_Q = asyncio.Queue()
    asyncio.create_task(broadcaster())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    await websocket.send_json(_state_ev())
    if SESSION:
        await websocket.send_json({"type": "backlog",
                                   "finals": SESSION.finals, "questions": SESSION.questions})
    try:
        while True:
            msg = await websocket.receive_json()
            t = msg.get("type")
            if t == "start":
                await start_session(msg)
            elif t == "stop":
                await stop_session()
            elif t == "set_provider":
                if SESSION:
                    SESSION.provider = msg.get("provider", SESSION.provider)
                    await _broadcast(_state_ev())
            elif t == "summarize":
                await summarize()
    except WebSocketDisconnect:
        CLIENTS.discard(websocket)
    except Exception:
        CLIENTS.discard(websocket)


def main():
    TRANSCRIPTS.mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{PORT}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Transcriber → {url}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
