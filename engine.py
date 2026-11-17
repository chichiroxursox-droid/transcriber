"""ASR engine — parakeet-mlx streaming (default) + segmented / Groq fallback.

The engine runs in ONE thread (the sole MLX caller). It drains raw s16le 16k mono
PCM from per-source queues, writes each source's WAV, feeds parakeet streaming
contexts, and emits transcript events via a thread-safe `emit(dict)` callback.

Two streaming contexts (mic="you", system="them") share one loaded model. That is
only safe because add_audio calls are serialized on this single thread and both
contexts use the same context_size (they set the encoder's attention mode
identically). M1 --selftest Test B proves this empirically before we rely on it.
"""
from __future__ import annotations

import os
import time
import wave
import queue
import threading
from datetime import datetime

# OpenBLAS on this macOS/Python build can deadlock during its threadpool init at
# numpy import time; pin it to one thread. MUST be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import mlx.core as mx
import parakeet_mlx

MODEL_NAME = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
SR = 16000
CTX = (256, 256)                 # streaming attention window (left history, lookahead)
CHUNK_SAMPLES = SR               # feed ~1s of audio per add_audio
BYTES_PER_CHUNK = CHUNK_SAMPLES * 2
FINAL_MAX_CHARS = 250            # force-cut a final even without punctuation past this
SENTENCE_END = ".?!"
SILENCE_SEC = 1.2                # gap of quiet that ends an utterance → flush + rotate
VOICE_THRESH = 300.0            # int16 RMS above this counts as speech (not room tone)
# Noise gate: 1s chunks quieter than this are NOT transcribed (room tone / faint
# background chatter). Raise it (400-500) to suppress louder background at the risk
# of clipping a soft primary speaker; lower it if your own voice gets dropped.
NOISE_GATE = float(os.environ.get("NOISE_GATE", 250))


def _rms(pcm_i16: np.ndarray) -> float:
    if len(pcm_i16) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))

_MODEL = None


def get_model(emit=None):
    """Load (once) and cache the parakeet model. First call downloads ~2.5GB."""
    global _MODEL
    if _MODEL is None:
        if emit:
            emit({"type": "status", "msg": f"loading model {MODEL_NAME}…"})
        t0 = time.time()
        _MODEL = parakeet_mlx.from_pretrained(MODEL_NAME)
        if emit:
            emit({"type": "status", "msg": f"model ready ({time.time()-t0:.1f}s)"})
    return _MODEL


def pcm16_to_mx(buf) -> mx.array:
    """int16 PCM bytes/ndarray → mlx float32 array in [-1, 1]."""
    a = np.frombuffer(buf, dtype=np.int16) if isinstance(buf, (bytes, bytearray)) else buf
    return mx.array(a.astype(np.float32) / 32768.0)


# One persistent worker thread owns MLX for the whole process. MLX is thread-affine
# (the model must be used on the thread it was loaded on), so every session must run
# on this same thread — spawning a fresh thread per session crashes the 2nd session.
_JOBS: "queue.Queue" = queue.Queue()
_WORKER = None
_WORKER_LOCK = threading.Lock()


def _worker_loop():
    while True:
        session, emit = _JOBS.get()
        try:
            run_engine(session, emit)
        except Exception as e:
            emit({"type": "error", "msg": f"engine error: {str(e)[:180]}"})
        finally:
            done = getattr(session, "engine_done", None)
            if done is not None:
                done.set()


def submit(session, emit):
    """Queue a session to run on the single persistent engine thread."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = threading.Thread(target=_worker_loop, daemon=True, name="engine-worker")
            _WORKER.start()
    _JOBS.put((session, emit))


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Committer:
    """Turns a streaming context's tokens into clean, timestamped finals.

    Two emission paths, because parakeet's finalized_tokens lags ~lookahead frames
    behind draft_tokens (a short utterance can sit entirely in draft):
      • update(): during speech, commit whole SENTENCES out of finalized_tokens as
        they cross a boundary; keep a live grey "draft" line = uncommitted-finalized
        tail + volatile draft tokens.
      • flush_tail(): at an utterance's end (silence) or stop, commit everything
        remaining (uncommitted finalized + ALL draft) as one final and clear the
        draft line. The caller then rotates the stream so this text isn't re-emitted
        when finalized later catches up.
    """

    def __init__(self, source: str, emit):
        self.source = source
        self.emit = emit
        self.idx = 0            # count of finalized_tokens already emitted as finals
        self._last_draft = None
        self._flushed = False   # draft already committed via flush_tail (avoid re-emit)

    def _emit_draft(self, text):
        if text != self._last_draft:
            self.emit({"type": "draft", "source": self.source, "text": text})
            self._last_draft = text

    def _emit_final(self, pending, n):
        text = "".join(t.text for t in pending[:n]).strip()
        if text:
            self.emit({"type": "final", "source": self.source,
                       "text": text, "t": _now_hms()})
        self.idx += n

    def update(self, tr):
        self._flushed = False
        pending = tr.finalized_tokens[self.idx:]
        pending_text = "".join(t.text for t in pending)
        self._emit_draft((pending_text + "".join(t.text for t in tr.draft_tokens)).strip())

        stripped = pending_text.rstrip()
        cut = max((stripped.rfind(c) for c in SENTENCE_END), default=-1)
        if cut != -1:
            commit_upto = cut + 1
        elif len(pending_text) > FINAL_MAX_CHARS:
            commit_upto = len(pending_text)
        else:
            return
        acc, n = 0, 0
        for t in pending:
            acc += len(t.text); n += 1
            if acc >= commit_upto:
                break
        self._emit_final(pending, n)
        self._last_draft = None

    def flush_tail(self, tr):
        pending = tr.finalized_tokens[self.idx:]
        text = ("".join(t.text for t in pending)
                + "".join(t.text for t in tr.draft_tokens)).strip()
        if text:
            self.emit({"type": "final", "source": self.source,
                       "text": text, "t": _now_hms()})
        self.emit({"type": "draft", "source": self.source, "text": ""})
        self.idx = len(tr.finalized_tokens)
        self._last_draft = ""
        self._flushed = True

    def reset(self):
        self.idx = 0
        self._last_draft = None
        self._flushed = False

    def has_content(self, tr):
        return len(tr.finalized_tokens) > self.idx or (
            len(tr.draft_tokens) > 0 and not self._flushed)


def _open_wav(path):
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    return w


def run_engine(session, emit):
    """Engine thread entry (streaming mode). Drains session.queues, writes WAVs,
    feeds parakeet contexts, emits events. Returns when stop_flag set + queues dry.

    `session` must expose: sources (list of ("you"/"them", queue, wav_path)),
    stop_flag (threading.Event).
    """
    if getattr(session, "engine", "parakeet") == "groq":
        return _run_segmented(session, emit)

    model = get_model(emit)
    sources = session.sources

    def new_stream():
        s = model.transcribe_stream(context_size=CTX)
        s.__enter__()
        return s

    streams, committers, bufs, wavs, last_voice = {}, {}, {}, {}, {}
    for src, _q, wav_path in sources:
        streams[src] = new_stream()
        committers[src] = Committer(src, emit)
        bufs[src] = bytearray()
        wavs[src] = _open_wav(wav_path)
        last_voice[src] = time.time()

    emit({"type": "status", "msg": "listening"})
    try:
        while True:
            stopping = session.stop_flag.is_set()
            idle = True
            for src, q, _w in sources:
                got = bytearray()
                try:
                    while True:
                        got += q.get_nowait()
                except queue.Empty:
                    pass
                if got:
                    wavs[src].writeframes(bytes(got))
                    bufs[src] += got
                    ev = len(got) - (len(got) & 1)
                    if _rms(np.frombuffer(bytes(got[:ev]), dtype=np.int16)) > VOICE_THRESH:
                        last_voice[src] = time.time()

                buf = bufs[src]
                if len(buf) >= BYTES_PER_CHUNK or (stopping and len(buf) >= 2):
                    n = len(buf) - (len(buf) % 2)
                    chunk = bytes(buf[:n])
                    del buf[:n]
                    # noise gate: only transcribe chunks with real energy; skip room
                    # tone / faint background so it isn't transcribed as "You"
                    if _rms(np.frombuffer(chunk, dtype=np.int16)) >= NOISE_GATE:
                        streams[src].add_audio(pcm16_to_mx(chunk))
                        committers[src].update(streams[src])
                    idle = False
                if stopping and len(buf) < 2:
                    buf.clear()          # drop sub-sample tail so the drain check completes

                # utterance boundary: quiet for SILENCE_SEC (or stopping) + nothing buffered
                quiet = (time.time() - last_voice[src]) > SILENCE_SEC
                if len(buf) == 0 and (quiet or stopping) and committers[src].has_content(streams[src]):
                    committers[src].flush_tail(streams[src])
                    if not stopping:
                        streams[src].__exit__(None, None, None)
                        streams[src] = new_stream()
                        committers[src].reset()
                    idle = False

            if stopping and all(len(bufs[s]) == 0 for s, _q, _w in sources):
                break
            if idle:
                time.sleep(0.05)
    finally:
        for src in list(streams):
            try:
                streams[src].__exit__(None, None, None)
            except Exception:
                pass
            wavs[src].close()
    emit({"type": "status", "msg": "engine stopped"})


# ---------------------------------------------------------------------------
# Segmented fallback (used if streaming dual-context fails M1 Test B, and the
# Groq path). ~RMS energy gate → per-utterance transcribe.
# ---------------------------------------------------------------------------
class EnergyGate:
    """Emit closed speech segments from a 16k int16 stream via RMS gating.
    open at threshold, close after hangover of silence, min length, force-cut."""

    def __init__(self, threshold=350.0, hangover=0.7, min_len=0.3, max_len=10.0):
        self.th, self.hang, self.min_len, self.max_len = threshold, hangover, min_len, max_len
        self.frame = int(SR * 0.1)  # 100ms frames
        self._pending = np.zeros(0, dtype=np.int16)
        self._seg = []              # list of int16 frames while open
        self._open = False
        self._silence = 0.0
        self._dur = 0.0

    def feed(self, pcm_i16, force=False):
        """Feed int16 samples; return list of closed segments (np.int16)."""
        self._pending = np.concatenate([self._pending, np.asarray(pcm_i16, dtype=np.int16)])
        out = []
        while len(self._pending) >= self.frame:
            fr, self._pending = self._pending[:self.frame], self._pending[self.frame:]
            rms = float(np.sqrt(np.mean(fr.astype(np.float32) ** 2)))
            if rms >= self.th:
                self._open = True
                self._silence = 0.0
            elif self._open:
                self._silence += 0.1
            if self._open:
                self._seg.append(fr)
                self._dur += 0.1
                if self._silence >= self.hang or self._dur >= self.max_len:
                    out.append(self._close())
        if force and self._open:
            out.append(self._close())
        return [s for s in out if s is not None]

    def _close(self):
        seg = np.concatenate(self._seg) if self._seg else np.zeros(0, np.int16)
        self._seg, self._open, self._silence, self._dur = [], False, 0.0, 0.0
        return seg if len(seg) >= int(SR * self.min_len) else None


def _run_segmented(session, emit):
    """Groq fallback engine: RMS-gate each source into utterances, transcribe each
    via Groq whisper-large-v3-turbo, emit one final per utterance (no live drafts)."""
    from groq import Groq
    client = Groq()
    sources = session.sources
    gates = {src: EnergyGate() for src, _q, _w in sources}
    wavs = {src: _open_wav(w) for src, _q, w in sources}
    emit({"type": "status", "msg": "listening (groq)"})
    try:
        while True:
            stopping = session.stop_flag.is_set()
            idle = True
            for src, q, _w in sources:
                got = bytearray()
                try:
                    while True:
                        got += q.get_nowait()
                except queue.Empty:
                    pass
                if got:
                    wavs[src].writeframes(bytes(got))
                samples = np.frombuffer(bytes(got), dtype=np.int16) if got else np.zeros(0, np.int16)
                for seg in gates[src].feed(samples, force=stopping):
                    try:
                        txt = groq_transcribe(seg, client)
                    except Exception as e:
                        emit({"type": "error", "msg": f"groq: {str(e)[:140]}"}); continue
                    if txt:
                        emit({"type": "final", "source": src, "text": txt, "t": _now_hms()})
                    idle = False
                if got:
                    idle = False
            if stopping and all(q.empty() for _s, q, _w in sources):
                break
            if idle:
                time.sleep(0.05)
    finally:
        for wh in wavs.values():
            wh.close()
    emit({"type": "status", "msg": "engine stopped"})


def groq_transcribe(pcm_i16, client) -> str:
    """Transcribe one int16 16k segment via Groq whisper-large-v3-turbo."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.asarray(pcm_i16, dtype=np.int16).tobytes())
    buf.seek(0)
    r = client.audio.transcriptions.create(
        file=("seg.wav", buf.read()), model="whisper-large-v3-turbo",
        response_format="text", language="en")
    return (r if isinstance(r, str) else getattr(r, "text", "")).strip()


# ---------------------------------------------------------------------------
# M1 self-test: GO/NO-GO for the streaming architecture. No mic/audio hardware.
# ---------------------------------------------------------------------------
def _read_wav_i16(path) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _selftest():
    A, B = ".tmp/a.wav", ".tmp/b.wav"
    a_kw = ["astronom", "galaxy", "telescope"]
    b_kw = ["grandmother", "cinnamon", "strawberry"]
    model = get_model(lambda e: print("·", e.get("msg", "")))
    a, b = _read_wav_i16(A), _read_wav_i16(B)

    def collect(tr):
        return "".join(t.text for t in tr.finalized_tokens + tr.draft_tokens).lower()

    print("\n=== Test A: single stream ===")
    with model.transcribe_stream(context_size=CTX) as tr:
        for i in range(0, len(a), CHUNK_SAMPLES):
            tr.add_audio(pcm16_to_mx(a[i:i + CHUNK_SAMPLES]))
        txt = collect(tr)
    print("  transcript:", txt.strip())
    a_hits = [k for k in a_kw if k in txt]
    assert len(a_hits) >= 2, f"Test A FAIL: only matched {a_hits}"
    print(f"  ✓ matched {a_hits}")

    print("\n=== Test B: TWO contexts on one model, interleaved (GO/NO-GO) ===")
    with model.transcribe_stream(context_size=CTX) as ta, \
         model.transcribe_stream(context_size=CTX) as tb:
        n = max(len(a), len(b))
        for i in range(0, n, CHUNK_SAMPLES):
            if i < len(a):
                ta.add_audio(pcm16_to_mx(a[i:i + CHUNK_SAMPLES]))
            if i < len(b):
                tb.add_audio(pcm16_to_mx(b[i:i + CHUNK_SAMPLES]))
        ta_txt, tb_txt = collect(ta), collect(tb)
    print("  stream A:", ta_txt.strip())
    print("  stream B:", tb_txt.strip())
    a_ok = sum(k in ta_txt for k in a_kw) >= 2 and not any(k in ta_txt for k in b_kw)
    b_ok = sum(k in tb_txt for k in b_kw) >= 2 and not any(k in tb_txt for k in a_kw)
    assert a_ok, f"Test B FAIL: stream A contaminated/incomplete: {ta_txt!r}"
    assert b_ok, f"Test B FAIL: stream B contaminated/incomplete: {tb_txt!r}"
    print("  ✓ streams stayed separate — GO: dual-stream streaming mode is safe")
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
