# Transcriber

Live, on-device transcription for macOS that writes your voice and the meeting audio as two separate speakers, saves to markdown and a Google Doc, and suggests follow-up questions while people talk.

## Why

Meeting transcribers either send audio to a cloud service or guess who is speaking with a diarization model that merges voices. Transcriber runs the speech model locally on Apple Silicon and gets speaker separation for free: your microphone is "You", the computer's audio output is "Them", and they never share a stream.

## How it works

- **Two speakers by capture source, not by voice.** The mic (sounddevice) and system audio each push 16 kHz mono PCM into their own queue. Each queue feeds its own streaming ASR context, so there is no diarization step and no failure mode where two voices blend.
- **System audio through a Core Audio process tap.** "Them" comes from [AudioTee](https://github.com/makeusabrew/audiotee), which uses the macOS 14.2+ process-tap API. The tap reads audio before it reaches the output device, so it keeps working with headphones on, and the volume keys keep working. Wearing headphones also stops the meeting from leaking into the mic. A BlackHole loopback device is kept as a fallback (`SYSTEM_AUDIO=blackhole`).
- **parakeet-mlx streaming, with three fixes the library needed in practice** (details in [`tasks/lessons.md`](tasks/lessons.md)):
  - *Short utterances never finalize.* The model's committed tokens lag its draft tail by about 20 s of audio, so a short sentence can sit in draft forever. The engine detects the end of an utterance by audio energy (1.2 s below an RMS threshold, since a live tap streams bytes even during silence), commits the remaining finalized and draft tokens as one line, then discards and reopens the stream so that text is not emitted twice. Rotation also bounds memory on long sessions.
  - *Two contexts, one model.* Both speakers share a single loaded model. That is only safe because every `add_audio` call runs on one thread and both contexts use the same attention window. `engine.py --selftest` proves the two transcripts stay separate before the app relies on it.
  - *MLX is thread-affine.* A model is bound to the thread that loaded it, so a new thread per session crashed the second recording. One persistent engine worker owns MLX for the life of the process and sessions are queued to it.
- **One door between threads and the event loop.** The mic callback, AudioTee reader threads and engine worker all call `emit()`, which hands events to a single asyncio consumer that updates session state and fans out to WebSocket clients. No locks around session state; start and stop are serialized by one `asyncio.Lock`.
- **Durable output first.** `transcript.md` is written every 10 s and is the source of truth. The Google Doc (created per session through the `gws` Google Workspace CLI) is best-effort and turns itself off after 3 consecutive failures instead of blocking the transcript.
- **Live questions, two providers.** Every 20 s with new speech, Claude Haiku proposes 2-3 follow-up questions grounded in the transcript. `claude_cli` shells out to `claude -p` and strips `ANTHROPIC_API_KEY` from its environment so it bills the Claude Code subscription instead of the API key; `anthropic` calls the API directly and is much faster.
- **Recovering a stale audio stack.** In a server left up for days, PortAudio's CoreAudio binding goes stale (`-9986`). Opening an input retries, and re-initializes PortAudio only when no other stream is live.
- **Noise gate.** One-second chunks below an RMS floor (`NOISE_GATE`) are not transcribed, so room tone is not written as speech.

Built with Claude Code. The finished app went through a multi-agent code review that found 11 bugs (a broadcaster crash on concurrent tab close, start/stop races, an odd-length PCM buffer that could loop forever, leaked subprocesses on timeout), all fixed and re-tested.

## Stack

Python 3.10+ (developed on 3.14), FastAPI + uvicorn WebSockets, [parakeet-mlx](https://pypi.org/project/parakeet-mlx/) with NVIDIA's `parakeet-tdt-0.6b-v2` on MLX, sounddevice, AudioTee (Swift), one static HTML page with no build step. Optional: Anthropic API or the `claude` CLI for questions and summaries, Groq Whisper as a cloud ASR fallback, `gws` CLI for Google Docs.

## Run it locally

Needs an Apple Silicon Mac on macOS 14.2+, Python 3.10+ (macOS's built-in 3.9 is too old; use Homebrew), and the Xcode command line tools for Swift.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# AudioTee (third-party, MIT), pinned to the commit this was built against
git clone https://github.com/makeusabrew/audiotee vendor/audiotee
git -C vendor/audiotee checkout 56ac954
(cd vendor/audiotee && swift build -c release && codesign --force --sign - .build/release/audiotee)

cp .env.example .env     # optional: API keys and overrides
./run.sh                 # opens http://127.0.0.1:8737
```

The first run downloads the parakeet model (about 2.5 GB) from Hugging Face. macOS asks once for **Microphone** and **System Audio Recording** permission for the app that launched the server (grant them from Terminal.app if your editor does not show the prompt). AudioTee must be codesigned, even ad-hoc as above, or the System Audio Recording prompt may never appear.

Type a session name, pick sources, press Record. Each session writes `transcripts/<date>_<time>_<name>/` with `transcript.md`, `you.wav`, `them.wav` and `meta.json`. After stopping, Summarize appends bullet notes to the markdown and the Doc.

Options live in `.env` (see `.env.example`). Set `QUESTION_PROVIDER=claude_cli` to use a Claude Code subscription instead of API credits.

### Tests

The tests use two synthesized-speech fixtures. Generate them with macOS `say`:

```bash
mkdir -p .tmp
say -o .tmp/a.aiff "The astronomer measured the distant galaxy using a large telescope on the mountain."
say -o .tmp/b.aiff "My grandmother baked cinnamon bread and strawberry muffins in her kitchen this morning."
for f in a b; do afconvert -f WAVE -d LEI16@16000 -c 1 .tmp/$f.aiff .tmp/$f.wav; done

.venv/bin/python engine.py --selftest   # one stream, then two interleaved streams on one model
.venv/bin/python test_e2e.py            # real server + WebSocket client, replayed audio, Doc stubbed
.venv/bin/python test_live.py           # real AudioTee capture: plays a fixture through the speakers
```

## Status and limits

- In regular personal use since July 2026. Single user, local only: the server binds to 127.0.0.1 and has no auth.
- macOS on Apple Silicon only (MLX and Core Audio taps).
- Separation is by source. Two people in the same room both land in "You", and every remote participant is "Them".
- Without headphones, speaker audio reaches the mic and shows up in both columns. The tap also captures notification sounds, so turn on Do Not Disturb.
- `claude_cli` questions lag about 30 s because the CLI boots its full runtime per call. The API path takes about 2 s.
- The Groq fallback has no live draft text; it transcribes each closed utterance.
- The Google Doc sink depends on an authenticated `gws` CLI. Without it the app reports the error and keeps writing markdown.
- `run.sh` prefers a venv at `~/.venvs/transcriber` if one exists. A venv inside an iCloud-synced folder can be evicted to the cloud and make startup hang for hours (see `tasks/lessons.md`).
- Recording laws vary. Some places require every party's consent. Tell people you are transcribing.

## Credits

- [AudioTee](https://github.com/makeusabrew/audiotee) by Nick Payne, MIT License. Not included in this repo; fetched and built as above.
- [parakeet-mlx](https://pypi.org/project/parakeet-mlx/) (Apache-2.0) and NVIDIA's [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model (CC-BY-4.0), via the `mlx-community` conversion.
- [MLX](https://github.com/ml-explore/mlx) by Apple (MIT).
