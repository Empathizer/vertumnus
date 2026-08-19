"""Picks the compute device for RVC inference: always the NVIDIA GPU when
present (cuda:0) — never the Intel iGPU, which CUDA can't see anyway. Falls
back to MPS (Apple Silicon, for Mac dev/testing) or CPU.

Note: this app's real-time architecture runs inference on a worker thread
separate from the audio callback (see audio/pipeline.py). Calling into
torch/faiss (both OpenMP-linked) from a second thread was found to segfault
reliably on macOS — not an MPS-specific issue (it reproduced on CPU too);
it's an OpenMP thread-pool race that only shows up once a second thread
touches these libraries. The actual fix is OMP_NUM_THREADS=1, set process-
wide in every entry point (server/ws_server.py, run_rvc_test.py) before
numpy/torch/faiss are imported. Do not remove that without re-verifying
threaded inference is still crash-free.
"""
import platform

import torch


def select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
