#!/bin/bash
# Launch Transcriber (opens http://127.0.0.1:8737 in your browser)
export OPENBLAS_NUM_THREADS=1   # prevent numpy/OpenBLAS import deadlock on this Python build
cd "$(dirname "$0")"
# ponytail: project lives on iCloud-synced ~/Desktop, where evicted ("dataless") venv
# files re-download at ~76 KB/s and startup takes hours. Prefer a venv on local disk.
VENV="$HOME/.venvs/transcriber/bin/python"
[ -x "$VENV" ] || VENV=".venv/bin/python"
exec "$VENV" app.py
