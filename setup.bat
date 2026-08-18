@echo off
REM ============================================================
REM Vertumnus - one-command Windows environment setup.
REM
REM Installs every prerequisite we hit real conflicts with during
REM development (Python 3.10, Node, Rust, ffmpeg), then runs the
REM backend Python setup and frontend npm install.
REM
REM What this CANNOT safely automate (needs your explicit consent,
REM shows a driver-install prompt, sometimes wants a reboot):
REM   - VB-Cable (virtual microphone) - install manually, see below.
REM   - NVIDIA driver - update manually via GeForce Experience if
REM     check_gpu.py later reports it's too old.
REM
REM IMPORTANT: this has not been run on real Windows hardware yet
REM (built on macOS, no Windows machine available to test against).
REM Every step below either succeeded when we hit the equivalent
REM problem on Mac, or is a best-effort winget install with a clear
REM manual fallback if the package ID is wrong/unavailable. If a
REM step fails, read the fallback instructions it prints rather than
REM assuming the whole script is broken - most likely just that one
REM package name needs adjusting for winget's current catalog.
REM ============================================================
setlocal enabledelayedexpansion

where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: winget not found. It ships with Windows 11 by default -
    echo if missing, install "App Installer" from the Microsoft Store,
    echo then re-run this script.
    exit /b 1
)

echo === Checking Python 3.10 ===
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 not found - installing via winget...
    winget install --id Python.Python.3.10 -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget install of Python 3.10 failed. Install manually
        echo from https://www.python.org/downloads/release/python-31011/
        echo ^(check "Add python.exe to PATH" during install^), then re-run this script.
        exit /b 1
    )
    echo Python 3.10 installed. Close and reopen this terminal so PATH picks
    echo it up, then re-run this script to continue.
    exit /b 0
) else (
    echo Python 3.10 already present.
)

echo === Checking Rust ===
where cargo >nul 2>&1
if errorlevel 1 (
    echo Rust not found - installing via winget...
    winget install --id Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget install of Rust failed. Install manually from
        echo https://rustup.rs, then re-run this script.
        exit /b 1
    )
    REM The winget rustup package sometimes only installs the rustup
    REM bootstrapper without a default toolchain - cargo/rustc won't exist
    REM until a toolchain is actually installed.
    where rustup >nul 2>&1
    if not errorlevel 1 (
        rustup default stable
    )
    where cargo >nul 2>&1
    if errorlevel 1 (
        echo Rust tool installed, but cargo/rustc still aren't on PATH.
        echo Close and reopen this terminal, then run "rustup default stable"
        echo if cargo is still missing after that, then re-run this script.
        exit /b 0
    )
    echo Rust installed and ready.
) else (
    echo Rust already present.
)

echo === Checking Node.js ===
where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js not found - installing via winget...
    winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget install of Node.js failed. Install manually from
        echo https://nodejs.org ^(LTS version^), then re-run this script.
        exit /b 1
    )
    echo Node.js installed. Close and reopen this terminal for PATH changes,
    echo then re-run this script to continue.
    exit /b 0
) else (
    echo Node.js already present.
)

echo === Checking ffmpeg ===
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ffmpeg not found - installing via winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget install of ffmpeg failed. Install manually from
        echo https://ffmpeg.org/download.html and add it to PATH, then re-run.
        echo This only blocks mp3/m4a/ogg file-input testing, not the core app -
        echo continuing setup.
    ) else (
        echo ffmpeg installed. You may need to reopen this terminal for PATH
        echo changes to take effect.
    )
) else (
    echo ffmpeg already present.
)

echo.
echo === Setting up Python backend (venv, cu128 torch, dependencies) ===
call backend\setup.bat
if errorlevel 1 (
    echo Backend setup reported an error - see output above. Fix and re-run
    echo this script ^(it's safe to re-run; each step checks before acting^).
    exit /b 1
)

echo.
echo === Installing frontend dependencies ===
cd frontend
call npm install
if errorlevel 1 (
    echo npm install failed - see output above.
    cd ..
    exit /b 1
)
cd ..

echo.
echo ============================================================
echo Automated setup complete. Two things still need YOUR action:
echo.
echo 1. VIRTUAL MICROPHONE (required, can't be silently automated -
echo    it installs a system audio driver and shows a Windows
echo    security prompt):
echo    Download and install VB-Cable from https://vb-audio.com/Cable/
echo.
echo 2. GPU CHECK - confirm check_gpu.py printed True + "RTX 5060 Ti"
echo    + "Matmul succeeded" above. If the matmul crashed instead,
echo    switch to the nightly cu128 line ^(see backend\setup.bat,
echo    it's commented directly below the stable line^) and re-run
echo    backend\setup.bat.
echo.
echo Once both are done, run the app:
echo   Terminal 1: cd backend ^&^& venv\Scripts\activate.bat ^&^& python -m server.ws_server
echo   Terminal 2: cd frontend ^&^& npm run tauri dev
echo ============================================================
endlocal
