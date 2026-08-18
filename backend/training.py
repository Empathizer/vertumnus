"""
In-app voice model training: drives the separate rvc_training/ tool (its own
venv, own dependency set — deliberately isolated from the real-time
inference backend's venv) through the same steps we validated manually:
preprocess -> F0 extraction -> feature extraction -> train -> build index ->
copy the result into backend/models/ so it shows up in the app's voice list.

Runs in a background thread; reports progress via a callback so the
websocket server can broadcast it live. Training is CPU/GPU-heavy and can
take anywhere from minutes (small dataset, few epochs, GPU) to hours (CPU) —
this module only orchestrates it, it doesn't make it fast.
"""
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
TRAINING_TOOL_DIR = PROJECT_ROOT / "rvc_training"
# venv layout differs by platform: Windows uses Scripts\python.exe, POSIX
# uses bin/python3.10 (or bin/python — 3.10 explicitly since that's the
# only version this whole project supports).
if platform.system() == "Windows":
    TRAINING_VENV_PYTHON = TRAINING_TOOL_DIR / "venv" / "Scripts" / "python.exe"
else:
    TRAINING_VENV_PYTHON = TRAINING_TOOL_DIR / "venv" / "bin" / "python3.10"
MODELS_DIR = BACKEND_DIR / "models"

SAMPLE_RATE = 40000  # matches the pretrained_v2 base checkpoints we downloaded
VERSION = "v2"


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
    stage: str  # "preprocess" | "f0" | "features" | "train" | "index" | "done" | "error"
    message: str
    fraction: float  # 0..1, best-effort within the current stage


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    if not safe:
        raise TrainingError("Voice name must contain at least one letter/number.")
    return safe


def _check_prereqs() -> None:
    if not TRAINING_VENV_PYTHON.is_file():
        raise TrainingError(
            f"Training tool venv not found at {TRAINING_VENV_PYTHON}. "
            f"Run rvc_training/venv setup first (see README)."
        )
    for fname in ("f0G40k.pth", "f0D40k.pth"):
        if not (TRAINING_TOOL_DIR / "assets" / "pretrained_v2" / fname).is_file():
            raise TrainingError(f"Missing pretrained base checkpoint: {fname} in rvc_training/assets/pretrained_v2/")
    if not (TRAINING_TOOL_DIR / "assets" / "hubert" / "hubert_base.pt").is_file():
        raise TrainingError("Missing rvc_training/assets/hubert/hubert_base.pt")




def _build_filelist_and_config(exp_dir: Path, exp_name: str) -> None:
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / "3_feature768"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = (
        {p.stem for p in gt_wavs_dir.glob("*")}
        & {p.stem for p in feature_dir.glob("*")}
        & {p.stem.replace(".wav", "") for p in f0_dir.glob("*")}
        & {p.stem.replace(".wav", "") for p in f0nsf_dir.glob("*")}
    )
    if not names:
        raise TrainingError(
            "No matching preprocessed segments across gt_wavs/features/f0 — "
            "the source recording may be too short or silent."
        )

    # Use Path joining throughout, not manual "/" string concatenation —
    # mixing a Path's native separator (backslash on Windows) with a
    # hardcoded "/" produces mixed-separator paths like
    # "C:\...\0_gt_wavs/name.wav", which is fragile across the various
    # libraries (numpy, soundfile, this training tool's own loader) that
    # read these filelist lines back.
    lines = []
    for name in names:
        lines.append(
            f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|"
            f"{f0_dir / (name + '.wav.npy')}|{f0nsf_dir / (name + '.wav.npy')}|0"
        )
    mute_dir = TRAINING_TOOL_DIR / "logs" / "mute"
    for _ in range(2):
        lines.append(
            f"{mute_dir / '0_gt_wavs' / 'mute40k.wav'}|{mute_dir / '3_feature768' / 'mute.npy'}|"
            f"{mute_dir / '2a_f0' / 'mute.wav.npy'}|{mute_dir / '2b-f0nsf' / 'mute.wav.npy'}|0"
        )
    import random
    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines))

    config_src = TRAINING_TOOL_DIR / "configs" / "v1" / "40k.json"  # matches infer-web.py's own logic for 40k
    config = json.loads(config_src.read_text())
    (exp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=4, sort_keys=True))


def _build_index(exp_dir: Path, exp_name: str) -> Path:
    feature_dir = exp_dir / "3_feature768"
    npys = [np.load(p) for p in sorted(feature_dir.glob("*.npy"))]
    big_npy = np.concatenate(npys, axis=0)
    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]

    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i : i + 8192])

    out_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{VERSION}.index"
    faiss.write_index(index, str(out_path))
    return out_path


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
        """Kills the currently-running step's whole process tree (train.py
        spawns its own child worker processes via multiprocessing — killing
        just the top process would orphan those). os.killpg/getpgid are
        POSIX-only (don't exist on Windows at all — calling them there
        raises AttributeError), so this branches per platform."""
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

    def _run(self, source_file_path: str, voice_name: str, epochs: int, on_progress) -> None:
        try:
            _check_prereqs()
            exp_name = _sanitize_name(voice_name)
            source_path = Path(source_file_path)
            if not source_path.is_file():
                raise TrainingError(f"Source file not found: {source_file_path}")

            input_dir = TRAINING_TOOL_DIR / "training_input" / exp_name
            input_dir.mkdir(parents=True, exist_ok=True)
            for old in input_dir.glob("*"):
                old.unlink()
            shutil.copy(source_path, input_dir / source_path.name)

            exp_dir = TRAINING_TOOL_DIR / "logs" / exp_name
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            exp_dir.mkdir(parents=True)

            py = str(TRAINING_VENV_PYTHON)

            self._emit(on_progress, "preprocess", "Slicing and normalizing audio...", 0.0)
            self._run_step(
                [py, "infer/modules/train/preprocess.py", str(input_dir), str(SAMPLE_RATE), "4",
                 str(exp_dir), "False", "3.7"],
                cwd=TRAINING_TOOL_DIR,
            )

            self._emit(on_progress, "f0", "Extracting pitch (F0)...", 0.0)
            self._run_step(
                [py, "infer/modules/train/extract/extract_f0_print.py", str(exp_dir), "4", "harvest"],
                cwd=TRAINING_TOOL_DIR,
            )

            self._emit(on_progress, "features", "Extracting HuBERT content features...", 0.0)
            self._run_step(
                [py, "infer/modules/train/extract_feature_print.py", "cpu", "1", "0", "0", str(exp_dir), VERSION],
                cwd=TRAINING_TOOL_DIR,
            )

            self._emit(on_progress, "train", "Preparing training config...", 0.0)
            _build_filelist_and_config(exp_dir, exp_name)

            self._emit(on_progress, "train", f"Training (0/{epochs} epochs)...", 0.0)
            self._run_training(exp_dir, exp_name, epochs, on_progress)

            self._emit(on_progress, "index", "Building retrieval index...", 0.0)
            index_path = _build_index(exp_dir, exp_name)

            weights_src = TRAINING_TOOL_DIR / "assets" / "weights" / f"{exp_name}.pth"
            if not weights_src.is_file():
                raise TrainingError(f"Training finished but no extracted checkpoint found at {weights_src}")

            MODELS_DIR.mkdir(exist_ok=True)
            shutil.copy(weights_src, MODELS_DIR / f"{exp_name}.pth")
            shutil.copy(index_path, MODELS_DIR / f"{exp_name}.index")

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

    def _run_training(self, exp_dir: Path, exp_name: str, epochs: int, on_progress) -> None:
        self._check_cancelled()
        py = str(TRAINING_VENV_PYTHON)
        cmd = [
            py, "infer/modules/train/train.py",
            "-e", exp_name, "-sr", "40k", "-f0", "1", "-bs", "4",
            "-te", str(epochs), "-se", str(epochs),
            "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
            "-l", "1", "-c", "0", "-sw", "0", "-v", VERSION,
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(TRAINING_TOOL_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            **_popen_extra_kwargs(),
        )
        self._current_proc = proc
        epoch_re = re.compile(r"Epoch:\s*(\d+)")
        last_lines: list[str] = []
        for line in proc.stdout:
            last_lines.append(line)
            if len(last_lines) > 60:
                last_lines.pop(0)
            m = epoch_re.search(line)
            if m:
                epoch = int(m.group(1))
                self._emit(
                    on_progress, "train", f"Training ({epoch}/{epochs} epochs)...",
                    min(epoch / epochs, 0.99),
                )
        proc.wait()
        self._current_proc = None
        if self._cancelled:
            raise TrainingCancelled()
        if proc.returncode != 0:
            raise TrainingError(f"Training process failed:\n{''.join(last_lines[-40:])}")
