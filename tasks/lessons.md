# Lessons — Transcriber

## parakeet-mlx streaming: short utterances stay in `draft_tokens` forever
`StreamingParakeet` splits output into `finalized_tokens` (committed) and
`draft_tokens` (volatile tail). The finalized boundary lags the draft by
`context_size[1] * depth` encoder frames — with `(256, 256)` that's ~20s. So an
utterance shorter than the lookahead **never reaches `finalized_tokens`**; the
sentence-ending period lives in draft indefinitely and no sentence-boundary
commit fires. **Fix:** detect utterance end by audio energy (silence gap), then
`flush_tail` = commit `finalized[idx:] + draft_tokens` as one final and **rotate
the stream** (discard + reopen) so the flushed text isn't re-emitted when finalized
later catches up. Rotation also bounds long-session memory.

## Silence must be detected by RMS, not by queue inactivity
A live mic / audiotee streams **continuous** bytes even during silence (quiet
samples), so "no data arrived" never triggers. Track `last_voice` from per-chunk
RMS > threshold instead.

## Two `transcribe_stream` contexts can share one model — if single-threaded
`__enter__`/`__exit__` toggle a **shared** encoder attention mode. Safe only when
all `add_audio` calls are serialized on one thread and both use the same
`context_size`. Stream rotation is safe because exit+reopen is atomic (no other
`add_audio` runs between them). Proven by engine.py `--selftest` Test B.

## MLX is thread-affine — one persistent engine worker, not a thread per session
`from_pretrained` binds the model to the thread it's used on. Spawning a fresh
engine thread per session crashes the 2nd session with `RuntimeError: no
Stream(cpu, 1) in current thread`. Fix: a single long-lived worker thread
(`engine.submit` + `_worker_loop`) owns MLX for the whole process; every session
runs on it. Single-session-per-process tests never catch this — test Stop→Record.

## Cleanup: never `rm -rf` by broad pattern over a dir with user data
`find transcripts -name "*untitled*" -exec rm -rf` deleted the user's real
"Untitled" recordings alongside my test folders — sessions default to the
"untitled" slug, so test noise and real data collide. Transcript TEXT survived in
the per-session Google Doc, but local WAVs were lost. Look at the target before
deleting; only remove dirs you created (track their exact names), never a pattern.

## Startup "hang" was iCloud eviction, NOT the OpenBLAS deadlock (corrected Oct 2026)
Symptom (seen Sep 14, Sep 21, Oct 21): server launches but never serves — process
alive at ~0% CPU, tiny RSS, no output, `sample <pid>` shows 100% of samples blocked
in `read` (libsystem_kernel) deep in an import chain. For a month this was recorded
as a numpy/OpenBLAS threadpool deadlock. **That diagnosis was wrong.**

Real cause: this project lives under iCloud-synced `~/Desktop`, and macOS
"Optimize Mac Storage" had **evicted** the venv to the cloud. Evicted files carry
the `dataless` flag (`ls -lO` / `find -flags +dataless`); reading one silently
blocks on an on-demand iCloud download. Measured Oct 21, 2026:
- 7.8 MB `.so` from `.venv` → **76 KB/s** (102 s)
- 2.5 MB from `/usr` (local volume) → **347 MB/s**
- `static/index.html`, 16 KB → **7 m 23 s** (~36 B/s)
- `fileproviderd` pegged at 64–91% CPU throughout
- 10,621 of ~15,400 files in `.venv` were dataless → ~26 h to materialize

Parallelism does NOT help: 24 concurrent readers materialized 10 files in 90 s.
iCloud serializes on-demand hydration, so you cannot brute-force it.

Fix: put the venv on local disk, outside iCloud. `~/.venvs/transcriber`, rebuilt
from PyPI in 3 min (PyPI pulls at 5 MB/s — 66x faster than iCloud). Cold import of
numpy+mlx+parakeet went from hours to **7.5 s**. `run.sh` prefers that venv and
falls back to `./.venv`. The iCloud `.venv` was left in place, untouched.

What survives from the old lesson: `OPENBLAS_NUM_THREADS=1` did measurably help on
Sep 14 (numpy imported in <1 s once warm vs. appearing to hang), so the
`os.environ.setdefault` calls in engine.py/audio.py stay — but it was treating a
symptom that eviction was mostly causing. Don't credit it for fixing startup.

Debug tips that DID pay off: `sample <pid>` to see the stuck stack (blocked in
`read` ⇒ suspect storage, not locks); `ls -lO` to spot `dataless`; compare read
throughput against a known-local path like `/usr` before blaming the library; and
`pkill -9 -f app.py` between attempts, since frozen processes stack up and hold the
port while returning HTTP 000 (one had been wedged for 7 days).

**Scope: this is not a Transcriber bug.** 359,697 of 546,105 files under
`~/Desktop/CLAUDE CODE/` are dataless (319k in `node_modules`, 10k in `.venv`,
8k in `.git`). Any project there can hang the same way on first build/import.

## Environment
- parakeet-mlx/mlx need Python ≥3.10 — Apple's system `python3` is 3.9. Use a venv
  on Homebrew python3.14 (mlx ships cp314 arm64 wheels).
- `uvicorn[standard]` is required for WebSockets (bare uvicorn rejects the upgrade).
- audiotee must be **codesigned** (even ad-hoc) or the System Audio Recording TCC
  prompt may not fire. Its stdout is clean s16le PCM; stderr is JSON logs.
- python-dotenv strips inline `# comments` correctly.
- `claude -p` is ~30s/call (spins up the full runtime) — fine for a 20s+ question
  loop, too slow for anything tighter. Anthropic API is ~2s but needs credits.
