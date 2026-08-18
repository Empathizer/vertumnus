@echo off
REM Vertumnus backend setup - Windows 11, Python 3.10.11, CUDA 12.8 (sm_120 / RTX 5060 Ti)
setlocal enabledelayedexpansion

REM Always operate relative to this script's own location, regardless of
REM where it's invoked from (e.g. called from the root setup.bat via
REM "call backend\setup.bat" without a preceding cd).
cd /d "%~dp0"

echo === Checking for Python 3.10 launcher ===
py -3.10 --version
if errorlevel 1 (
    echo ERROR: Python 3.10 not found via "py -3.10". Install Python 3.10.11 from python.org
    echo and make sure the "py" launcher is on PATH, then re-run this script.
    exit /b 1
)

echo === Creating venv ===
py -3.10 -m venv venv
if errorlevel 1 (
    echo ERROR: venv creation failed.
    exit /b 1
)

call venv\Scripts\activate.bat

echo === Setting pip version ===
REM fairseq==0.12.2 depends on omegaconf<2.1, whose published wheels have
REM malformed version metadata that pip>=24.1 refuses to resolve. Pin below
REM that line rather than using the latest pip.
python -m pip install "pip<24.1"

echo === Installing remaining backend dependencies ===
REM Installed BEFORE torch on purpose: fairseq/torchcrepe both declare torch
REM as their own dependency, and pip's resolver doesn't know our upcoming
REM cu128 install is special — installing torch first and requirements.txt
REM second let pip silently pull in a default CPU-only torch afterward and
REM overwrite the cu128 build (confirmed on real hardware: check_gpu.py
REM reported "torch version: 2.13.0+cpu" despite the cu128 line running
REM without error earlier in the script). Torch is installed LAST instead,
REM below, so it's always the final, authoritative install.
pip install -r requirements.txt

echo === Installing PyTorch (cu128 STABLE) ===
echo NOTE: sm_120 (RTX 5060 Ti / Blackwell) requires a cu128 build of torch 2.7+.
echo Do NOT install torch from requirements.txt.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM If the stable line above does not yet support sm_120 on your system, comment the
REM stable install above and uncomment the nightly line below instead:
REM pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo === Running GPU check ===
python check_gpu.py

echo.
echo Setup complete. If check_gpu.py did NOT print True + "NVIDIA GeForce RTX 5060 Ti"
echo and a successful matmul, see the instructions it printed (nightly fallback).
endlocal
