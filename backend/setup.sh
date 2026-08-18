#!/usr/bin/env bash
# Vertumnus backend setup - macOS (dev/testing machine, Apple Silicon or Intel).
# No NVIDIA GPU here: torch runs on MPS (Apple Silicon) or CPU.
# The RTX 5060 Ti / CUDA cu128 path is Windows-only — see setup.bat.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "=== python3.10 not found, installing via Homebrew ==="
    if ! command -v brew >/dev/null 2>&1; then
        echo "ERROR: Homebrew not found. Install it from https://brew.sh, then re-run."
        exit 1
    fi
    brew install python@3.10
fi

echo "=== Creating venv ==="
python3.10 -m venv venv

# shellcheck disable=SC1091
source venv/bin/activate

echo "=== Setting pip version ==="
# fairseq==0.12.2 depends on omegaconf<2.1, whose published wheels have
# malformed version metadata that pip>=24.1 refuses to resolve. Pin below
# that line rather than using the latest pip.
python -m pip install "pip<24.1"

echo "=== Installing PyTorch (standard build: MPS/CPU, no CUDA on Mac) ==="
pip install torch torchaudio

echo "=== Installing remaining backend dependencies ==="
pip install -r requirements.txt

echo "=== Running GPU/device check ==="
python check_gpu.py

echo
echo "Setup complete. This validates the Mac dev path only (MPS/CPU)."
echo "The CUDA/RTX 5060 Ti path must still be verified on Windows via setup.bat."
