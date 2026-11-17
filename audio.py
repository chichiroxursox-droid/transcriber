"""Audio capture: mic ("you") and system audio ("them"), two independent 16k
mono s16le streams. Each source pushes raw PCM bytes into its queue and emits
throttled level events. No mixing, no aggregate device.

System audio uses audiotee (Core Audio process tap, headphone-proof). Set
SYSTEM_AUDIO=blackhole to instead read the BlackHole 2ch input device.
"""
from __future__ import annotations

import os
import time
import queue
import threading
import subprocess
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # avoid numpy/OpenBLAS import deadlock

import numpy as np
import sounddevice as sd

SR = 16000
AUDIOTEE = str(Path(__file__).parent / "vendor/audiotee/.build/release/audiotee")


def _level(pcm_i16: np.ndarray) -> float:
    if len(pcm_i16) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(pcm_i16.astype(np.float32) ** 2)))
    return min(1.0, rms / 3000.0)


_OPEN = set()        # streams _open_input handed out that are still live


def _open_input(**kw):
    """Open an InputStream, retrying CoreAudio failures and rebuilding PortAudio.

    Two failure modes seen in the wild, both fatal to a session start:
      -10851 "Invalid Property Value" — transient, while the device list is
        reshuffled (an iPhone Continuity mic connecting, or the previous
        session's stream not yet released). A plain retry clears it.
      -9986  paInternalError — persistent *within one process*. PortAudio binds
        to CoreAudio once at import; in a server left up for days (sleep/wake,
        devices coming and going) that binding goes stale, so every open fails
        while a fresh process opens the same device fine. Only re-initialising
        PortAudio fixes it without a restart.

    The reinit tears down every PortAudio stream in the process, so it only runs
    when we hold none — otherwise it would silently kill a live capture.
    ponytail: 3 tries covers both; if a device genuinely cannot do SR, resample
    from its native rate rather than raising the try count.
    """
    last = None
    for attempt in range(3):
        if attempt and not _OPEN:
            try:
                sd._terminate(); sd._initialize()
            except Exception:
                pass                      # reinit is best-effort; keep retrying
        s = None
        try:
            s = sd.InputStream(**kw)
            s.start()
            _OPEN.add(s)
            return s
        except Exception as e:
            last = e
            if s is not None:
                try: s.close()
                except Exception: pass
            time.sleep(0.3 * (attempt + 1))
    raise last


def _close_input(stream):
    """Stop/close a stream from _open_input and drop it from the live set."""
    _OPEN.discard(stream)
    try:
        stream.stop(); stream.close()
    except Exception:
        pass


class MicSource:
    """Microphone → q (16k mono int16 bytes) via sounddevice."""

    def __init__(self, q: queue.Queue, emit, device=None):
        self.q, self.emit = q, emit
        self.device = device if device is not None else os.environ.get("MIC_DEVICE")
        if self.device is not None and str(self.device).isdigit():
            self.device = int(self.device)
        self._stream = None
        self._n = 0

    def _cb(self, indata, frames, tinfo, status):
        self.q.put(indata.tobytes())
        self._n += 1
        if self._n % 4 == 0:                       # ~throttle level to a few Hz
            self.emit({"type": "level", "source": "you", "rms": _level(indata[:, 0])})

    def start(self):
        self._stream = _open_input(
            samplerate=SR, channels=1, dtype="int16",
            device=self.device, blocksize=int(SR * 0.1), callback=self._cb)

    def stop(self):
        if self._stream:
            _close_input(self._stream); self._stream = None


class SystemSource:
    """System audio → q. audiotee subprocess (default) or BlackHole device."""

    def __init__(self, q: queue.Queue, emit, mode: str | None = None):
        self.q, self.emit = q, emit
        self.mode = mode or os.environ.get("SYSTEM_AUDIO", "audiotee")
        self._proc = None
        self._stream = None
        self._threads = []
        self._run = False
        self._n = 0

    # --- audiotee path ---
    def _read_stdout(self):
        chunk = int(SR * 0.1) * 2                   # 0.1s of s16le
        total = 0
        while self._run and self._proc and self._proc.stdout:
            buf = self._proc.stdout.read(chunk)
            if not buf:
                break
            total += len(buf)
            self.q.put(buf)
            self._n += 1
            if self._n % 4 == 0:
                even = len(buf) - (len(buf) & 1)     # read() may return an odd count
                a = np.frombuffer(buf[:even], dtype=np.int16)
                self.emit({"type": "level", "source": "them", "rms": _level(a)})
        # EOF: if audiotee exited having produced no audio, the tap was likely denied
        if self._run and total == 0:
            rc = self._proc.poll() if self._proc else None
            self.emit({"type": "error", "msg":
                       "no system audio captured — grant 'System Audio Recording' in "
                       f"System Settings ▸ Privacy (audiotee exited {rc})"})

    def _read_stderr(self):
        while self._run and self._proc and self._proc.stderr:
            line = self._proc.stderr.readline()
            if not line:
                break
            txt = line.decode(errors="replace").strip()
            low = txt.lower()
            if "error" in low or "fail" in low or "denied" in low:
                self.emit({"type": "error", "msg": f"system audio: {txt[:160]}"})

    # --- blackhole path ---
    def _bh_cb(self, indata, frames, tinfo, status):
        mono = indata[:, 0].copy()                  # BlackHole is 2ch; take L
        self.q.put(mono.tobytes())
        self._n += 1
        if self._n % 4 == 0:
            self.emit({"type": "level", "source": "them", "rms": _level(mono)})

    def start(self):
        self._run = True
        if self.mode == "blackhole":
            dev = next((i for i, d in enumerate(sd.query_devices())
                        if "BlackHole" in d["name"] and d["max_input_channels"] > 0), None)
            if dev is None:
                self.emit({"type": "error", "msg": "BlackHole 2ch input not found"})
                return
            self._stream = _open_input(
                samplerate=SR, channels=2, dtype="int16", device=dev,
                blocksize=int(SR * 0.1), callback=self._bh_cb)
        else:
            if not Path(AUDIOTEE).exists():
                self.emit({"type": "error", "msg": f"audiotee not built at {AUDIOTEE}"})
                return
            self._proc = subprocess.Popen(
                [AUDIOTEE, "--sample-rate", str(SR)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            for fn in (self._read_stdout, self._read_stderr):
                t = threading.Thread(target=fn, daemon=True); t.start()
                self._threads.append(t)

    def stop(self):
        self._run = False
        if self._stream:
            _close_input(self._stream); self._stream = None
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


# ---------------------------------------------------------------------------
# M2 / M3 test mains — drive the REAL engine.run_engine so we test the true path.
# ---------------------------------------------------------------------------
class _TestSession:
    def __init__(self, sources):
        self.sources = sources
        self.stop_flag = threading.Event()


def _run_test(which: str):
    import engine
    Path(".tmp").mkdir(exist_ok=True)
    q = queue.Queue()
    emit = lambda ev: (print(f"[{ev.get('source','')}] {ev['text']}")
                       if ev["type"] == "final" else
                       (print("STATUS:", ev.get("msg")) if ev["type"] in ("status", "error") else None))
    if which == "mic":
        src = MicSource(q, emit); label, wav = "you", ".tmp/you.wav"
    else:
        src = SystemSource(q, emit); label, wav = "them", ".tmp/them.wav"
    session = _TestSession([(label, q, wav)])
    src.start()
    print(f"--- {which} test running. Speak / play audio. Ctrl-C to stop. ---")
    th = threading.Thread(target=engine.run_engine, args=(session, emit)); th.start()
    try:
        while th.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
        src.stop(); session.stop_flag.set(); th.join(timeout=15)
        print(f"wrote {wav}")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    if "--mic-test" in sys.argv:
        _run_test("mic")
    elif "--system-test" in sys.argv:
        _run_test("system")
    else:
        print("usage: python audio.py --mic-test | --system-test")
