# Vertumnus

Real-time desktop voice changer: mic in, converted voice out, to a virtual
microphone any app (Discord, Zoom, WhatsApp Desktop, OBS, games) can pick up.

Backend: Python (PyTorch + RVC voice conversion + pedalboard DSP). Frontend:
Tauri (Rust) + React + TypeScript, talking to the backend over a local
websocket (`ws://127.0.0.1:8765`).

## 0. Windows quickstart (one command)

From the project root, in an ordinary (non-admin) terminal:

```
setup.bat
```

This installs everything it can via `winget` (Python 3.10, Rust, Node.js,
ffmpeg — only whatever's actually missing), then runs the backend Python
setup (venv, cu128 torch, dependencies, `check_gpu.py`) and `npm install`
for the frontend. It's safe to re-run — every step checks before acting.
It may ask you to reopen the terminal once (so a freshly-installed tool's
PATH takes effect) and re-run itself; just follow what it prints.

**Not automated on purpose** (both need your explicit consent/interaction):
- **VB-Cable** (virtual microphone) — download and install from
  vb-audio.com/Cable, then reboot. Silently installing a system audio
  driver isn't something a setup script should do without you seeing it.
- **NVIDIA driver** — update via GeForce Experience or nvidia.com if
  `check_gpu.py` reports it's too old.

This was built and validated on macOS (no Windows machine was available
during development) — each step either mirrors a fix validated the hard
way on Mac, or is a best-effort `winget` install with a printed fallback if
that package ID doesn't resolve. If a step fails, read what it printed
rather than assuming the whole script is broken.

Once `setup.bat` finishes and both manual items above are done, skip to
[Running](#4-running).

## 1. Install prerequisites (manual path / macOS)

### Windows 11 (primary target — NVIDIA GPU accelerated)

Prefer `setup.bat` above. Manually, you need:

1. **NVIDIA driver 580.82+** — required for CUDA 12.8. Update via GeForce
   Experience or nvidia.com if needed.
2. **VB-Cable** (virtual microphone) — download and install from
   vb-audio.com/Cable, then reboot. This creates a "CABLE Input" playback
   device (what this app outputs to) and a "CABLE Output" recording device
   (what Discord/Zoom/etc. should select as their microphone).
3. **Python 3.10.11** — install from python.org, and make sure the `py`
   launcher is on PATH (the installer does this by default). Do not use
   3.11/3.12 — RVC's dependencies (fairseq) need 3.10.
4. **Node.js** (LTS) and **Rust** (via rustup.rs) — needed to build the
   Tauri frontend shell.
5. **ffmpeg** — on PATH, needed to decode mp3/m4a/ogg files (test-audio
   playback and voice-model training source files). `winget install
   Gyan.FFmpeg` or download from ffmpeg.org.

### macOS (dev/testing machine — CPU/MPS only, no NVIDIA GPU)

1. **BlackHole** (virtual microphone): `brew install blackhole-2ch`, then
   reboot. If audio through it plays at the wrong speed/pitch, check
   BlackHole's nominal sample rate in Audio MIDI Setup — the app resamples
   to whatever rate it reports, but a stale/unexpected rate there is worth
   ruling out first.
2. **Python 3.10**: `brew install python@3.10`
3. **ffmpeg**: `brew install ffmpeg` — needed to decode mp3/m4a/ogg files.
4. Node.js and Rust as above, if building the frontend here too.

Voice conversion is much slower on macOS since there's no CUDA — expect
higher latency there. Treat Mac as a dev/UI-testing target, not the
performance target.

## 2. Backend setup

```
cd backend
./setup.sh        # macOS
setup.bat         # Windows
```

This creates a `venv` with Python 3.10, installs PyTorch **separately** from
`requirements.txt` (never put torch in requirements.txt — it needs a specific
index URL per platform), installs the rest of the dependencies, and runs
`check_gpu.py`.

**Windows** installs the CUDA 12.8 (cu128) build, required for the RTX 5060
Ti's Blackwell (sm_120) architecture — standard pip wheels (cu118/cu121) do
NOT include sm_120 kernels and will crash with "no kernel image is available
for execution on the device":

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If `check_gpu.py`'s CUDA matmul crashes with that error, switch to the
nightly cu128 line (commented in `setup.bat`):

```
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

The CUDA runtime + cuDNN ship bundled inside the torch wheel — do not install
a separate CUDA toolkit.

**Confirm before continuing to voice conversion:** `check_gpu.py` should
print `torch.cuda.is_available(): True`, a device name containing
`NVIDIA GeForce RTX 5060 Ti`, and "Matmul succeeded."

## 3. Voice models

This app does not bundle any pretrained voices. You need:

1. **HuBERT content encoder** (required for all RVC models — a generic
   speech feature extractor, not a voice): download `hubert_base.pt` from
   https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
   and place it at `backend/models/hubert_base.pt`.
2. **Your voice model**: an RVC `.pth` file you trained yourself or that is
   licensed for your use, optionally with a matching `.index` file (same
   filename stem, e.g. `myvoice.pth` + `myvoice.index`). Drop both into
   `backend/models/`, or train one directly from the app:

   In the running app's **Train a new voice** section: Browse to any
   recording on disk (needs your rights to that voice), give it a name, set
   an epoch count (typical good-quality range is 200-300 for a ~5-10 minute
   recording — far more practical on the RTX box than CPU), and click
   **Start Training**. This drives the separate `rvc_training/` tool (its
   own venv — see below) through preprocessing, pitch/feature extraction,
   training, and index building, with live progress, and the result appears
   automatically in the Voice model dropdown when done — no manual file
   copying. First time only, that tool needs its own one-time setup:
   ```
   git clone --branch 2.2.231006 --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc_training
   cd rvc_training
   py -3.10 -m venv venv
   venv\Scripts\activate.bat
   python -m pip install "pip<24.1"
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
   Then download the pretrained base checkpoints and HuBERT into
   `rvc_training/assets/pretrained_v2/` (`f0G40k.pth`, `f0D40k.pth`) and
   `rvc_training/assets/hubert/hubert_base.pt` — same URLs as above, plus
   `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth`
   and `.../f0D40k.pth`.

   One known snag from developing this on Mac that may resurface on a fresh
   Windows clone: this training tool's `infer/lib/audio.py` uses an older
   PyAV API (`av.open(path, "rb")` and `add_stream(fmt, channels=1)`) that
   newer `av` wheels reject. If preprocessing fails with an `av`-related
   error, open that file and change `"rb"`/`"wb"` to `"r"`/`"w"`, and
   replace the `channels=1` argument to `add_stream` with a separate
   `ostream.layout = "mono"` line after it — same fix either platform.

Do not use models trained to clone a specific named real person without
their consent/license — that's out of scope for this app.

## 4. Running

Backend:

```
cd backend
source venv/bin/activate   # or venv\Scripts\activate.bat on Windows
python -m server.ws_server
```

This starts the websocket control server on `ws://127.0.0.1:8765` and picks
the compute device automatically — `cuda:0` on Windows with the RTX 5060 Ti,
MPS/CPU on macOS. It never falls back to the Intel iGPU (CUDA can't see it
anyway).

Frontend (separate terminal):

```
cd frontend
npm install
npm run tauri dev
```

First time only: Tauri needs app icons before it can build a bundle —
generate a placeholder set with `npx tauri icon path/to/any-1024x1024.png`
(any square PNG works for local dev/testing).

You can also test the backend headlessly before the frontend exists/works,
using the CLI scripts in `backend/`:

- `python run_passthrough.py` — mic → virtual mic, no processing.
- `python run_rvc_test.py` — mic → RVC conversion → virtual mic.

## 5. Using it in Discord/Zoom/OBS/etc.

1. Start the backend and frontend (or a CLI test script).
2. In the app, set **Input** to your real mic and **Output** to the virtual
   mic device (`CABLE Input` on Windows, `BlackHole 2ch` on macOS) — the UI
   auto-detects and pre-selects it if found.
3. Load a voice model, adjust effects, click **Start**.
4. In Discord/Zoom/WhatsApp/OBS/your game, open audio/microphone settings and
   select the **matching input side** of the virtual device (`CABLE Output`
   on Windows, `BlackHole 2ch` on macOS — same device, the "other end" of the
   virtual cable).
5. Enable the **Monitor** toggle in Vertumnus if you want to hear yourself
   while talking (it does not affect what other apps hear).

## 6. Building a Windows installer

From `frontend/`, with Rust and Node installed on the Windows machine:

```
npm install
npm run tauri build
```

This produces an NSIS installer under `frontend/src-tauri/target/release/bundle/nsis/`.
The bundled app still expects the Python backend to be running separately —
this build produces the frontend shell only; packaging the Python backend
into the installer (e.g. via PyInstaller, invoked as a sidecar binary) is a
follow-up step, not yet wired up.

## 7. Error states

The backend reports structured errors over the websocket
(`{"type": "error", "kind": ..., "message": ...}`) for every failure mode
called out as a hard requirement:

- `model_not_found` — bad `.pth`/`.index` path, or missing `hubert_base.pt`.
- `device_not_found` — invalid/unplugged input, output, or monitor device.
- `virtual_mic_not_installed` — no VB-Cable/BlackHole-like device detected.
- `cuda_unavailable` — CUDA not available to PyTorch (wrong wheel/driver).
- `cuda_kernel_mismatch` — the sm_120 kernel crash; switch to the nightly
  cu128 torch line.

The frontend surfaces each of these as a banner with a specific next step
(see `ERROR_HELP` in `frontend/src/App.tsx`).

## macOS differences (recap)

- No NVIDIA GPU — runs on MPS (Apple Silicon) or CPU, both noticeably slower
  than the RTX 5060 Ti for real-time conversion. Expect to raise `block_time`
  in the voice-load settings for a usable latency/quality tradeoff.
  Practically, mac is a UI/pass-through/DSP dev target — validate real-time
  voice conversion quality and latency on the Windows/RTX 5060 Ti build.
- Virtual mic is BlackHole instead of VB-Cable.
