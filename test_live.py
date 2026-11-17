"""Real-hardware E2E: boot the actual app with REAL audiotee capture (no replay),
start a system-audio session, play a known fixture through the speakers, and confirm
the tap → engine → WebSocket path yields the expected 'Them' transcript.

Proves M3 through the full app stack. gdoc stubbed to stay hermetic.
"""
import time
import json
import asyncio
import threading
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent / ".env"))

import uvicorn
import websockets
import app as appmod

PORT = 8801


async def drive():
    finals = []
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "start", "title": "live tap",
                                  "mic": False, "system": True, "engine": "parakeet"}))
        # wait for engine to be listening, then play the fixture through speakers
        await asyncio.sleep(4)
        subprocess.Popen(["afplay", ".tmp/a.wav"])
        t0 = time.time()
        while time.time() - t0 < 20 and not finals:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 2.0))
            except asyncio.TimeoutError:
                continue
            if m["type"] == "final":
                finals.append(m); print(f"  FINAL [{m['source']}] {m['text']}")
            elif m["type"] in ("status", "error"):
                print(f"  {m['type'].upper()}: {m.get('msg')}")
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(1.5)
    return finals


def main():
    async def _c(t): return "STUB", "https://docs.google.com/document/d/STUB/edit"
    async def _a(d, t): return None
    appmod.gdoc.create_doc = _c
    appmod.gdoc.append = _a

    server = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)

    finals = asyncio.run(drive())
    them = " ".join(f["text"].lower() for f in finals if f["source"] == "them")
    print(f"\n  them transcript: {them!r}")
    assert "astronom" in them or "telescope" in them or "galaxy" in them, \
        "real audiotee capture did not transcribe the fixture"
    print("\nLIVE TAP E2E PASSED ✓ — real system-audio capture works through the full app")


if __name__ == "__main__":
    main()
