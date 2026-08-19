# RVC training GPU crash — investigation summary (for continuing on the Windows PC)

## Setup
- Machine: Windows 11, RTX 5060 Ti 16GB (Blackwell, sm_120), Intel Ultra 9 285K, 32GB RAM
- Project root: `C:\Users\MSI\Desktop\vertumnus-for-windows`
- Separate training tool: `rvc_training\` (official RVC-Project repo, tag `2.2.231006`, its own venv)
- Goal: use the app's in-app "Train a new voice" feature (drives `rvc_training\infer\modules\train\train.py`) to train a custom voice from a short WhatsApp audio recording (experiment name `WhatsApp_Audio_2026-08-18_at_3_23_44_PM`, source dataset already preprocessed — features/F0/filelist already exist under `rvc_training\logs\WhatsApp_Audio_2026-08-18_at_3_23_44_PM\`, so training can be re-run without redoing preprocessing).

## Core finding
Training crashes every single time with a **native, silent Windows access violation** (`Windows fatal exception: access violation`) inside PyTorch's own CUDA backward pass:
```
Thread ... (most recent call first):
  File "...\torch\autograd\graph.py", line 882, in _engine_run_backward
  File "...\torch\autograd\__init__.py", line 379, in backward
  File "...\torch\_tensor.py", line 631, in backward
  File "...\rvc_training\infer\modules\train\train.py", line ~483/484, in train_and_evaluate
```
This crash only appears with `PYTHONFAULTHANDLER=1` set — without it, the process just dies silently with exit code 0 and no traceback, which made this very hard to diagnose (looked identical to a clean-but-premature exit for many attempts).

## Already fixed, real bugs (keep these — not the GPU crash, but genuinely broken on Windows)
Apply these first before doing anything else — they're required regardless of CPU/GPU:

1. **fairseq `torch.load` weights_only issue** (torch 2.6+ changed the default) — patched directly in the venv's site-packages:
   `venv\Lib\site-packages\fairseq\checkpoint_utils.py` line ~315:
   `torch.load(f, map_location=torch.device("cpu"))` → add `, weights_only=False`

2. **PyAV old API** in `rvc_training\infer\lib\audio.py` — `av.open(path, "rb")`/`"wb"` → `"r"`/`"w"`, and `add_stream(fmt, channels=1)` → `add_stream(fmt)` + separate `ostream.layout = "mono"` line. (Already documented in the main project's README as a known snag.)

3. **matplotlib API change** in `rvc_training\infer\lib\train\utils.py` (`plot_spectrogram_to_numpy` and similar) — old `fig.canvas.tostring_rgb()` → `np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)` then reshape to `(h, w, 4)` and slice `[:, :, :3]` to drop alpha.

4. **Windows DataLoader multiprocessing issues** in `rvc_training\infer\modules\train\train.py` (`train_loader = DataLoader(...)` around line 157-165):
   - `num_workers=4,` → `num_workers=0,`
   - `persistent_workers=True,` → `persistent_workers=False,`
   - `prefetch_factor=8,` → `prefetch_factor=None,`
   (These were needed to get past earlier `ValueError`s and silent early exits — genuinely necessary on Windows, unrelated to the GPU crash below.)

## What we ruled out for the GPU backward-pass crash
All of these were tried and **did not fix it** — the exact same crash, same location, every time:
- Stable `torch==2.11.0+cu128` (from `https://download.pytorch.org/whl/cu128`)
- Nightly `torch` (from `https://download.pytorch.org/whl/nightly/cu128`, fetched fresh same day)
- `fp16_run: True` vs manually forced `False` in `rvc_training\logs\<exp>\config.json` — no difference
- `torch.backends.cudnn.enabled = False` (forces non-cuDNN fallback conv kernels) — no difference
- Batch size 1 vs 4 — no difference
- NVIDIA driver already very current (610.62, CUDA UMD 13.3) — not a driver issue
- Basic CUDA ops work fine: plain matmul (`check_gpu.py`) passes, and STFT (`torch.stft`, used in mel-spectrogram loss) also runs fine — the crash is specifically in `.backward()`, not forward.

## Leading untested hypothesis (best next thing to try)
The model uses PyTorch's **old** `torch.nn.utils.weight_norm` API extensively (confirmed via repeated deprecation warnings: `torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm`, from `infer\lib\infer_pack\models.py` and wherever `WeightNorm.apply(module, name, dim)` is called). The old API has its own hand-written custom CUDA backward kernel (separate from ordinary autograd ops), which is a plausible candidate for a narrow, still-missing sm_120 kernel — the newer `parametrizations.weight_norm` decomposes into ordinary, already-supported ops instead.

**Next step:** find every place `torch.nn.utils.weight_norm(...)` is applied (likely `infer\lib\infer_pack\models.py`, `infer\lib\infer_pack\modules.py`, and similar) and every corresponding `nn.utils.remove_weight_norm(...)` call (used in model's own `remove_weight_norm()` methods, needed for final export), and switch both to the `torch.nn.utils.parametrizations.weight_norm` / `torch.nn.utils.parametrize.remove_parametrizations(module, name, leave_parametrized=True)` API consistently. This must be done carefully — the old and new APIs are not drop-in replacements, and getting the removal side wrong will break final checkpoint export (`.pth` extraction) even if training itself starts working.

If that doesn't fix it either, next things worth trying (not yet attempted):
- Pin a specific intermediate `torch` version instead of latest-stable or nightly (e.g. `torch==2.8.*+cu128` or `2.9.*+cu128`) — Blackwell kernel coverage has been added incrementally release-by-release, so a middle version could have this particular kernel where both the newest and nightly builds are missing it.
- `CUDA_LAUNCH_BLOCKING=1` combined with `PYTHONFAULTHANDLER=1` for a possibly more precise crash location.

## Confirmed working fallbacks (already proven end-to-end on this exact PC/dataset)
- **CPU training**: fully works after the 4 "already fixed" bugs above (do NOT apply the `.cuda()`-stripping patch unless deliberately forcing CPU — that's a separate, more invasive patch not needed for the GPU path). Confirmed one full epoch completed in ~6 min 10 sec on this CPU (Intel Ultra 9 285K) for the ~80-segment dataset. At that rate, 500 epochs ≈ 51 hours; a low-epoch run (e.g. `-te 25 -se 25`, ~25 min) is a practical way to get a rough model quickly.
- **Real-time voice conversion (inference)** already works perfectly on the RTX 5060 Ti via the main app — this GPU bug is training-only (backward pass), not inference (forward-only, matmul-heavy).

## Exact command used to reproduce/retest
```
venv\Scripts\python.exe infer\modules\train\train.py -e WhatsApp_Audio_2026-08-18_at_3_23_44_PM -sr 40k -f0 1 -bs 1 -te 500 -se 500 -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth -l 1 -c 0 -sw 0 -v v2
```
Run with `set PYTHONFAULTHANDLER=1` first to get the crash traceback if it recurs.
