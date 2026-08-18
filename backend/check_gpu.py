"""
Verifies the environment is correctly set up for accelerated inference before
any ML/RVC work happens.

Run: python check_gpu.py

On Windows (target deployment machine, RTX 5060 Ti):
  Expected: torch.cuda.is_available() == True, device name contains
  "RTX 5060 Ti", and a real CUDA matmul completes without error (proves
  sm_120 Blackwell kernels are present in the torch build).

On macOS (dev/testing machine, Apple Silicon, no NVIDIA GPU):
  There is no CUDA here. This just verifies torch is installed and runs a
  matmul on the best available device (MPS if present, else CPU). This is
  informational only — it does NOT prove anything about the Windows/CUDA path.
"""
import sys
import platform


def fail(message: str) -> None:
    print("\n" + "=" * 70)
    print("GPU CHECK FAILED")
    print("=" * 70)
    print(message)
    sys.exit(1)


def check_windows(torch) -> None:
    print(f"torch.version.cuda: {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")

    if not available:
        fail(
            "CUDA is not available to PyTorch.\n"
            "Checklist:\n"
            "  1. Confirm you installed the cu128 wheel, not a plain 'pip install torch'.\n"
            "  2. Confirm your NVIDIA driver is current (580.82+).\n"
            "  3. Re-run setup.bat.\n"
            "If it still fails, switch to the nightly cu128 line in setup.bat:\n"
            "  pip install --pre torch torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu128"
        )
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"torch.cuda.get_device_name(0): {device_name}")

    if "RTX 5060 Ti" not in device_name:
        print(
            f"WARNING: expected 'NVIDIA GeForce RTX 5060 Ti' but got "
            f"'{device_name}'. Make sure the NVIDIA GPU is selected, not the "
            f"Intel integrated GPU."
        )

    print("\nRunning CUDA matmul on cuda:0 to verify sm_120 kernels are present...")
    try:
        device = torch.device("cuda:0")
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        c = a @ b
        torch.cuda.synchronize()
        result_sum = c.sum().item()
        print(f"Matmul succeeded. Result checksum: {result_sum:.4f}")
    except RuntimeError as e:
        fail(
            f"CUDA matmul crashed: {e}\n\n"
            "This is the classic sm_120 kernel mismatch: your torch build does "
            "not include Blackwell (sm_120) kernels.\n"
            "Fix: switch to the NIGHTLY cu128 line.\n"
            "  1. pip uninstall torch torchaudio\n"
            "  2. pip install --pre torch torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu128\n"
            "  3. Re-run this script."
        )
        return

    print("\n" + "=" * 70)
    print("GPU CHECK PASSED — CUDA available, RTX 5060 Ti detected, sm_120 "
          "matmul succeeded.")
    print("=" * 70)


def check_macos(torch) -> None:
    print("Running on macOS — no NVIDIA GPU expected here. This is a "
          "dev/testing check only; it does not validate the CUDA path.")

    mps_available = getattr(torch.backends, "mps", None) is not None \
        and torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_available(): {mps_available}")

    device = torch.device("mps") if mps_available else torch.device("cpu")
    print(f"Using device: {device}")

    try:
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        c = a @ b
        if device.type == "mps":
            torch.mps.synchronize()
        result_sum = c.sum().item()
        print(f"Matmul succeeded on {device}. Result checksum: {result_sum:.4f}")
    except RuntimeError as e:
        fail(f"Matmul on {device} crashed: {e}")
        return

    print("\n" + "=" * 70)
    print(f"GPU CHECK PASSED (macOS, {device}) — torch works on this machine. "
          "Remember: the Windows/RTX 5060 Ti/CUDA path is unverified by this run.")
    print("=" * 70)


def main() -> None:
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python version: {platform.python_version()}")
    if not platform.python_version().startswith("3.10"):
        print("WARNING: expected Python 3.10.11 — RVC/fairseq compatibility "
              "is not guaranteed on other versions.")

    try:
        import torch
    except ImportError:
        fail(
            "PyTorch is not installed.\n"
            "Install it separately (never via requirements.txt).\n"
            "Windows (cu128): pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "macOS: pip install torch torchaudio"
        )
        return

    print(f"torch version: {torch.__version__}")

    if platform.system() == "Windows":
        check_windows(torch)
    else:
        check_macos(torch)


if __name__ == "__main__":
    main()
