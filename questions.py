"""Live suggested-question generation, two interchangeable providers.

- "anthropic":  claude-haiku-4-5 via the Anthropic API (needs ANTHROPIC_API_KEY)
- "claude_cli": headless `claude -p --model haiku` (uses the Claude Code subscription)

Ethan switches between them live in the UI to compare. generate() returns
(questions, elapsed_ms) so the UI can show per-refresh latency + provider.
"""
from __future__ import annotations

import os
import re
import time
import asyncio

API_MODEL = "claude-haiku-4-5-20251001"

SYSTEM = (
    "You are assisting someone who is listening to a live conversation or meeting. "
    "From the running transcript, propose the 2-3 sharpest follow-up questions THIS "
    "listener could ask next to go deeper, clarify a gap, or surface what's unsaid. "
    "Ground every question in something actually said. Make them specific and concise "
    "(one sentence each). Do not repeat questions already asked. "
    "Output ONLY the questions, one per line, no numbering, no preamble."
)

_client = None


def _anthropic():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic
        _client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _build_prompt(tail: str, existing: list[str]) -> str:
    prior = "\n".join(f"- {q}" for q in existing[-8:]) or "(none yet)"
    return (f"Transcript so far (most recent last):\n\"\"\"\n{tail[-4000:]}\n\"\"\"\n\n"
            f"Questions already suggested (do not repeat):\n{prior}\n\n"
            f"Give 2-3 new questions now.")


_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-•*])\s+")   # strip only a real list marker


def _parse(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = _MARKER.sub("", line.strip()).strip()
        if len(s) > 3:
            out.append(s)
    return out[:3]


async def _via_api(prompt: str) -> str:
    msg = await _anthropic().messages.create(
        model=API_MODEL, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


async def _via_cli(prompt: str) -> str:
    # Hide API-key auth so `claude` uses the claude.ai subscription login instead
    # (with ANTHROPIC_API_KEY set, the CLI prefers it and disables the subscription).
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", f"{SYSTEM}\n\n{prompt}", "--model", "haiku",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), 45.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()                 # reap; don't leak the claude process
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {err.decode()[:300]}")
    return out.decode()


async def generate(tail: str, existing: list[str], provider: str) -> tuple[list[str], int]:
    prompt = _build_prompt(tail, existing)
    t0 = time.time()
    raw = await (_via_cli(prompt) if provider == "claude_cli" else _via_api(prompt))
    return _parse(raw), int((time.time() - t0) * 1000)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tail = ("Alice: We shipped the new onboarding flow last week. "
            "Bob: Activation went up but week-two retention actually dropped a few points. "
            "Alice: Right, and we're not sure if it's the flow or the email sequence.")
    for prov in ("anthropic", "claude_cli"):
        qs, ms = asyncio.run(generate(tail, [], prov))
        print(f"\n=== {prov} ({ms}ms) ===")
        for q in qs:
            print(" •", q)
