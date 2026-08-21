"""
In-app voice model training: drives the separate ddsp_training/ tool (its own
venv, own dependency set — deliberately isolated from the real-time
inference backend's venv, mirroring how rvc_training/ was isolated) through
DDSP-SVC's rectified-flow pipeline: split train/val -> preprocess (pitch,
units, volume extraction) -> train -> copy the result into backend/models/
so it shows up in the app's voice list.

New training always goes through DDSP-SVC (not RVC) — RVC training crashes
reliably on this machine's Blackwell GPU during the discriminator's backward
pass (a genuine, unfixed PyTorch/CUDA kernel bug, confirmed across stable and
nightly builds); DDSP-SVC's rectified-flow objective has no discriminator
and doesn't touch that broken kernel. RVC's real-time inference path
(backend/rvc/, backend/ddsp_engine/) is unaffected and stays available for
existing/other RVC models.

Runs in a background thread; reports progress via a callback so the
websocket server can broadcast it live.
"""
import platform
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import os
import soundfile as sf
import yaml

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DDSP_TOOL_DIR = PROJECT_ROOT / "ddsp_training"
# venv layout differs by platform: Windows uses Scripts\python.exe, POSIX
# uses bin/python3.10 (or bin/python — 3.10 explicitly since that's the
# only version this whole project supports).
if platform.system() == "Windows":
    DDSP_VENV_PYTHON = DDSP_TOOL_DIR / "venv" / "Scripts" / "python.exe"
else:
    DDSP_VENV_PYTHON = DDSP_TOOL_DIR / "venv" / "bin" / "python3.10"
MODELS_DIR = BACKEND_DIR / "models"

DDSP_BASE_CONFIG = DDSP_TOOL_DIR / "configs" / "reflow.yaml"
DDSP_SAMPLE_RATE = 44100  # matches configs/reflow.yaml's data.sampling_rate

VAL_HOLDOUT_FRACTION = 0.10
VAL_HOLDOUT_MAX_SECONDS = 60.0
VAL_HOLDOUT_MIN_SECONDS = 6.0


class TrainingError(Exception):
    pass


class TrainingCancelled(Exception):
    pass


def _popen_extra_kwargs() -> dict:
    """start_new_session is POSIX-only (Python's own docs: "the setsid()
    system call will be made ... POSIX only"). taskkill /T (used in
    cancel() on Windows) walks the process tree by PID regardless of how
    it was spawned, so Windows doesn't need an equivalent flag here."""
    if platform.system() == "Windows":
        return {}
    return {"start_new_session": True}


@dataclass
class TrainingProgress:
    stage: str  # "preprocess" | "train" | "done" | "error"
    message: str
    fraction: float  # 0..1, best-effort within the current stage


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    if not safe:
        raise TrainingError("Voice name must contain at least one letter/number.")
    return safe


def _check_prereqs() -> None:
    if not DDSP_VENV_PYTHON.is_file():
        raise TrainingError(
            f"Training tool venv not found at {DDSP_VENV_PYTHON}. "
            f"Set up ddsp_training/venv first (see README)."
        )
    if not (DDSP_TOOL_DIR / "pretrain" / "contentvec" / "pytorch_model.bin").is_file():
        raise TrainingError("Missing ddsp_training/pretrain/contentvec/pytorch_model.bin")
    if not (DDSP_TOOL_DIR / "pretrain" / "rmvpe" / "model.pt").is_file():
        raise TrainingError("Missing ddsp_training/pretrain/rmvpe/model.pt")
    for fname in ("model", "config.json"):
        if not (DDSP_TOOL_DIR / "pretrain" / "nsf_hifigan" / fname).is_file():
            raise TrainingError(f"Missing ddsp_training/pretrain/nsf_hifigan/{fname}")


def _split_train_val(source_path: Path, work_dir: Path, sample_rate: int) -> tuple[Path, Path]:
    """Loads source_path and takes the LAST min(10%, 60s) (but at least 6s —
    below that, everything just goes to train and val gets whatever's left
    over) as validation, everything before that as train. Resamples to
    sample_rate, writes work_dir/train/audio/{stem}.wav and
    work_dir/val/audio/{stem}.wav.

    DDSP-SVC (unlike RVC) accepts whole long .wav files directly — no
    pre-slicing into short segments needed, its own AudioDataset samples
    random duration-second crops per training step."""
    audio, _ = librosa.load(str(source_path), sr=sample_rate, mono=True)
    total_seconds = len(audio) / sample_rate

    val_seconds = min(total_seconds * VAL_HOLDOUT_FRACTION, VAL_HOLDOUT_MAX_SECONDS)
    if val_seconds < VAL_HOLDOUT_MIN_SECONDS:
        val_seconds = min(VAL_HOLDOUT_MIN_SECONDS, total_seconds * 0.5)
    val_samples = max(1, int(val_seconds * sample_rate))

    train_audio = audio[:-val_samples] if val_samples < len(audio) else audio[:1]
    val_audio = audio[-val_samples:]

    train_dir = work_dir / "train" / "audio"
    val_dir = work_dir / "val" / "audio"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stem = source_path.stem
    train_path = train_dir / f"{stem}.wav"
    val_path = val_dir / f"{stem}.wav"
    sf.write(train_path, train_audio, sample_rate)
    sf.write(val_path, val_audio, sample_rate)
    return train_path, val_path


class TrainingManager:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self.is_training = False
        self._current_proc: subprocess.Popen | None = None
        self._cancelled = False

    def start(
        self,
        source_file_path: str,
        voice_name: str,
        epochs: int = 20,
        on_progress=None,
    ) -> None:
        if self.is_training:
            raise TrainingError("A training run is already in progress.")
        self.is_training = True
        self._cancelled = False
        self._thread = threading.Thread(
            target=self._run, args=(source_file_path, voice_name, epochs, on_progress), daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        """Kills the currently-running step's whole process tree (train_reflow.py
        spawns its own dataloader worker processes — killing just the top
        process would orphan those). os.killpg/getpgid are POSIX-only (don't
        exist on Windows at all — calling them there raises AttributeError),
        so this branches per platform."""
        self._cancelled = True
        if self._current_proc is None or self._current_proc.poll() is not None:
            return
        pid = self._current_proc.pid
        if platform.system() == "Windows":
            # taskkill /T walks and kills the whole descendant process tree
            # by PID, independent of how the process was spawned.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _emit(self, on_progress, stage: str, message: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(TrainingProgress(stage=stage, message=message, fraction=fraction))

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise TrainingCancelled()

    def _run_step(self, cmd: list[str], cwd: Path) -> None:
        self._check_cancelled()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, **_popen_extra_kwargs(),
        )
        self._current_proc = proc
        output_lines: list[str] = []
        for line in proc.stdout:
            output_lines.append(line)
            if len(output_lines) > 60:
                output_lines.pop(0)
        proc.wait()
        self._current_proc = None
        if self._cancelled:
            raise TrainingCancelled()
        if proc.returncode != 0:
            tail = "".join(output_lines[-40:])
            raise TrainingError(f"Command failed: {' '.join(cmd)}\n{tail}")

    def _write_generated_config(self, exp_name: str, epochs: int) -> Path:
        """Clones configs/reflow.yaml with data.train_path/valid_path/env.expdir
        rewritten to a fresh per-run location, and train.epochs/interval_val/
        interval_force_save rewritten for this run's target length — same
        role as RVC's old _build_filelist_and_config(), just YAML-key
        rewriting instead of filelist/index math. A per-run copy avoids
        racing/polluting the vendored default config across runs."""
        with open(DDSP_BASE_CONFIG, "r") as f:
            config = yaml.safe_load(f)

        data_dir = DDSP_TOOL_DIR / "data" / "_generated" / exp_name
        exp_dir = DDSP_TOOL_DIR / "exp" / exp_name

        config["data"]["train_path"] = str(data_dir / "train")
        config["data"]["valid_path"] = str(data_dir / "val")
        config["env"]["expdir"] = str(exp_dir)
        # Matches RVC's prior behavior (save_every_epoch == total_epoch: a
        # single checkpoint at the very end), kept simple and predictable.
        target = max(1, epochs)
        config["train"]["epochs"] = target
        config["train"]["interval_val"] = target
        config["train"]["interval_force_save"] = target

        generated_dir = DDSP_TOOL_DIR / "configs" / "_generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        config_path = generated_dir / f"{exp_name}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        return config_path

    def _run(self, source_file_path: str, voice_name: str, epochs: int, on_progress) -> None:
        try:
            _check_prereqs()
            exp_name = _sanitize_name(voice_name)
            source_path = Path(source_file_path)
            if not source_path.is_file():
                raise TrainingError(f"Source file not found: {source_file_path}")

            py = str(DDSP_VENV_PYTHON)

            self._emit(on_progress, "preprocess", "Splitting into train/validation clips...", 0.0)
            data_dir = DDSP_TOOL_DIR / "data" / "_generated" / exp_name
            if data_dir.exists():
                shutil.rmtree(data_dir)
            _split_train_val(source_path, data_dir, DDSP_SAMPLE_RATE)

            exp_dir = DDSP_TOOL_DIR / "exp" / exp_name
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            config_path = self._write_generated_config(exp_name, epochs)

            self._emit(on_progress, "preprocess", "Extracting pitch/units/volume features...", 0.05)
            self._run_step(
                [py, "preprocess.py", "-c", str(config_path)],
                cwd=DDSP_TOOL_DIR,
            )

            self._emit(on_progress, "train", f"Training (0/{epochs})...", 0.1)
            training_output = self._run_training(config_path, epochs, on_progress)

            # DDSP-SVC's train_reflow.py, same as RVC's train.py before it,
            # never checks its own exit code against what actually happened
            # inside — treat "no checkpoint produced" as the real failure
            # signal, and surface whatever the process printed (e.g. a
            # native crash traceback) instead of a bare file-missing message.
            checkpoints = sorted(
                exp_dir.glob("model_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
            )
            if not checkpoints:
                raise TrainingError(
                    "Training did not produce a checkpoint — it crashed or was "
                    f"killed before finishing (no model_*.pt in {exp_dir}). "
                    f"Last training output:\n{training_output}"
                )
            latest = checkpoints[-1]

            self._emit(on_progress, "train", "Copying trained model...", 0.95)
            dest_dir = MODELS_DIR / exp_name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True)
            shutil.copy(latest, dest_dir / "model.pt")
            shutil.copy(exp_dir / "config.yaml", dest_dir / "config.yaml")

            self._emit(on_progress, "done", f"Done — {exp_name} is ready to load.", 1.0)
        except TrainingCancelled:
            self._emit(on_progress, "error", "Training cancelled.", 0.0)
        except TrainingError as e:
            self._emit(on_progress, "error", str(e), 0.0)
        except Exception as e:
            self._emit(on_progress, "error", f"Unexpected error: {e}", 0.0)
        finally:
            self.is_training = False
            self._current_proc = None

    # Substrings that show up when the training subprocess dies from a native
    # crash (e.g. a CUDA kernel access violation) rather than a clean Python
    # exception. The vendored training tool's own exit code is useless for
    # detecting this (see the comment in _run), so output content is the only
    # signal available. Engine-agnostic — this exact mechanism was already
    # validated against the RVC crash this machine is prone to.
    _CRASH_SIGNATURES = (
        "Windows fatal exception",
        "Fatal Python error",
        "Segmentation fault",
    )

    def _run_training(self, config_path: Path, epochs: int, on_progress) -> str:
        self._check_cancelled()
        py = str(DDSP_VENV_PYTHON)
        cmd = [py, "train_reflow.py", "-c", str(config_path)]
        # PYTHONFAULTHANDLER makes a native crash print a low-level thread/stack
        # dump before the process dies, instead of vanishing with no diagnostic
        # output at all — that dump is what _CRASH_SIGNATURES looks for below.
        env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
        proc = subprocess.Popen(
            cmd, cwd=str(DDSP_TOOL_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            **_popen_extra_kwargs(),
        )
        self._current_proc = proc
        step_re = re.compile(r"step:\s*(\d+)")
        last_lines: list[str] = []
        # DDSP-SVC's dataset can yield more than one batch per "epoch" for a
        # real (non-trivial-length) training clip, so the step counter isn't
        # always a clean 1:1 match with the "epoch" count reflected in
        # train.epochs (confirmed empirically: step observed dropping back
        # down mid-run before climbing past the target again). Track the
        # running max so the reported fraction never visibly regresses, even
        # though the underlying training is progressing correctly either way.
        max_step_seen = 0
        for line in proc.stdout:
            last_lines.append(line)
            if len(last_lines) > 300:
                last_lines.pop(0)
            m = step_re.search(line)
            if m:
                max_step_seen = max(max_step_seen, int(m.group(1)))
                self._emit(
                    on_progress, "train", f"Training ({max_step_seen}/{epochs})...",
                    min(0.1 + 0.85 * max_step_seen / max(1, epochs), 0.99),
                )
        proc.wait()
        self._current_proc = None
        if self._cancelled:
            raise TrainingCancelled()
        tail = "".join(last_lines[-60:])
        if proc.returncode != 0:
            raise TrainingError(f"Training process failed:\n{tail}")
        if any(sig in "".join(last_lines) for sig in self._CRASH_SIGNATURES):
            raise TrainingError(
                "Training crashed (native error, likely a GPU/CUDA kernel "
                f"failure) even though the process reported success:\n{tail}"
            )
        return tail
