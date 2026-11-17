"""Hermetic end-to-end test: real uvicorn server + real websocket client, driving
the full app (WS → engine → session files) with replayed fixture WAVs. No mic/
speakers. gdoc is stubbed (proven separately) to stay hermetic and avoid Drive spam.
"""
import time
import json
import wave
import asyncio
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent / ".env"))

import uvicorn
import websockets
import audio
import app as appmod

PORT = 8799


class ReplaySource:
    """Push a fixture WAV into the queue in 0.1s chunks (~5x realtime), then idle."""
    def __init__(self, q, emit, wav, **_):
        self.q, self.wav, self._run = q, wav, False

    def start(self):
        self._run = True
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self):
        with wave.open(self.wav, "rb") as w:
            data = w.readframes(w.getnframes())
        step = int(16000 * 0.1) * 2
        for i in range(0, len(data), step):
            if not self._run:
                return
            self.q.put(data[i:i + step]); time.sleep(0.02)

    def stop(self):
        self._run = False


async def drive():
    finals, drafts, levels = [], 0, 0
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws:
        await ws.recv()  # initial state
        await ws.send(json.dumps({"type": "start", "title": "e2e probe",
                                  "mic": True, "system": True, "engine": "parakeet"}))
        t0 = time.time()
        while time.time() - t0 < 25 and len(finals) < 2:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 2.0))
            except asyncio.TimeoutError:
                continue
            if m["type"] == "final":
                finals.append(m); print(f"  FINAL [{m['source']}] {m['text']}")
            elif m["type"] == "draft":
                drafts += 1
            elif m["type"] == "level":
                levels += 1
            elif m["type"] in ("status", "error"):
                print(f"  {m['type'].upper()}: {m.get('msg')}")
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(1.5)
    return finals, drafts, levels


def main():
    audio.MicSource = lambda q, emit, **k: ReplaySource(q, emit, ".tmp/b.wav")      # you
    audio.SystemSource = lambda q, emit, **k: ReplaySource(q, emit, ".tmp/a.wav")   # them

    async def _c(title): return "STUB", "https://docs.google.com/document/d/STUB/edit"
    async def _a(did, text): return None
    appmod.gdoc.create_doc = _c
    appmod.gdoc.append = _a

    server = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)

    finals, drafts, levels = asyncio.run(drive())

    print(f"\n  finals={len(finals)} drafts={drafts} levels={levels}")
    srcs = {e["source"] for e in finals}
    assert "you" in srcs and "them" in srcs, f"missing a source: {srcs}"
    you = " ".join(e["text"].lower() for e in finals if e["source"] == "you")
    them = " ".join(e["text"].lower() for e in finals if e["source"] == "them")
    assert "grandmother" in you or "cinnamon" in you, f"you wrong: {you!r}"
    assert "astronom" in them or "telescope" in them, f"them wrong: {them!r}"
    assert drafts > 0, "no draft events streamed"

    sess = sorted(appmod.TRANSCRIPTS.glob("*_e2e-probe"))[-1]
    md = (sess / "transcript.md").read_text()
    assert "You:" in md and "Them:" in md, "transcript.md missing speaker lines"
    assert (sess / "meta.json").exists(), "meta.json missing"
    assert (sess / "you.wav").exists() and (sess / "them.wav").exists(), "wavs missing"
    print(f"  session: {sess.name}  (md {len(md)}b, meta ✓, wavs ✓)")
    print("\nE2E PASSED ✓")


if __name__ == "__main__":
    main()
