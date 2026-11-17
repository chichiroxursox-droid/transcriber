"""Google Doc sink via the authed `gws` CLI (argv lists — no shell quoting).

create_doc(title) -> (document_id, url); append(document_id, text).
Both are async (run gws as a subprocess off the event loop).
"""
from __future__ import annotations

import json
import asyncio


def _extract_json(stdout: str) -> dict:
    """gws may print a keyring notice before the JSON; parse from first brace."""
    i = stdout.find("{")
    if i == -1:
        raise ValueError(f"no JSON in gws output: {stdout[:200]!r}")
    return json.loads(stdout[i:])


async def _run(args: list[str], timeout: float = 20.0) -> str:
    proc = await asyncio.create_subprocess_exec(
        "gws", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()                 # reap so no zombie / transport warning
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"gws {args[0]} failed: {err.decode()[:300]}")
    return out.decode()


async def create_doc(title: str) -> tuple[str, str]:
    out = await _run(["docs", "documents", "create", "--json", json.dumps({"title": title})])
    doc_id = _extract_json(out)["documentId"]
    return doc_id, f"https://docs.google.com/document/d/{doc_id}/edit"


async def append(document_id: str, text: str) -> None:
    if not text:
        return
    await _run(["docs", "+write", "--document", document_id, "--text", text])


if __name__ == "__main__":
    async def _demo():
        did, url = await create_doc("Transcriber gdoc self-test")
        print("created:", url)
        await append(did, "Hello from gdoc.py self-test.\n")
        print("appended OK — check the doc")
    asyncio.run(_demo())
