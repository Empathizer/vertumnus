"""
Manual test: mic -> DDSP-SVC (rectified-flow) voice conversion -> virtual
mic, no UI. Mirrors run_rvc_test.py exactly, for the DDSP engine path.

Run: python run_ddsp_test.py
Requires: ddsp_training/pretrain/{contentvec,rmvpe,nsf_hifigan}/... (see
ddsp_training's own README), and at least one trained voice folder
(model.pt + config.yaml) dropped into backend/models/<name>/.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# See server/ws_server.py for why this is required (background-thread +
# OpenMP-linked library segfault, confirmed empirically on macOS).
os.environ.setdefault("OMP_NUM_THREADS", "1")

from audio.devices import list_devices, print_devices
from audio.pipeline import AudioPipeline, PipelineError, find_virtual_mic_device
from gpu_device import select_torch_device
from voice_models import list_voice_models


def prompt_index(label: str) -> int:
    valid = {d.index for d in list_devices()}
    while True:
        raw = input(f"{label}: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a device index number.")
            continue
        if idx not in valid:
            print("Not a valid device index.")
            continue
        return idx


def main() -> None:
    device = select_torch_device()
    print(f"Compute device: {device}")

    models = [m for m in list_voice_models() if m.engine == "ddsp"]
    if not models:
        print("No DDSP-SVC voice models found in backend/models/. Train one via "
              "the app, or drop a <name>/ folder containing model.pt + config.yaml "
              "there and re-run.")
        return

    print("\n=== DDSP-SVC voice models ===")
    for i, m in enumerate(models):
        print(f"  [{i}] {m.name}")
    choice = int(input("Select model index: ").strip())
    model = models[choice]

    print_devices()
    input_device = prompt_index("Select INPUT device index (your mic)")
    virtual_mic = find_virtual_mic_device()
    if virtual_mic is not None:
        print(f"Detected likely virtual mic: [{virtual_mic}]")
    output_device = prompt_index("Select OUTPUT device index (virtual mic)")

    pipeline = AudioPipeline(device)
    try:
        print("Loading voice model (ContentVec + RMVPE + NSF-HiFiGAN + reflow "
              "model, plus a one-time CUDA warm-up pass — takes a few seconds)...")
        pipeline.load_voice(model.pth_path, engine="ddsp", config_path=model.config_path)
    except PipelineError as e:
        print(f"\n[{e.kind}] {e.message}")
        return

    print(f"Model sample rate: {pipeline.voice.target_sample_rate} Hz, "
          f"block size: {pipeline.converter.block_frame} samples, "
          f"extra latency from SOLA crossfade: {pipeline.converter.latency_seconds() * 1000:.0f} ms")

    try:
        pipeline.start(input_device, output_device)
    except PipelineError as e:
        print(f"\n[{e.kind}] {e.message}")
        return

    print("\nRunning. Speak into your mic. Latency prints every 2s. Press Enter to stop.\n")

    import threading

    stop_event = threading.Event()

    def print_latency():
        while not stop_event.is_set():
            print(f"model={pipeline.latency.model_ms:.1f}ms total={pipeline.latency.total_ms:.1f}ms "
                  f"underruns={pipeline.underrun_count}")
            stop_event.wait(2.0)

    threading.Thread(target=print_latency, daemon=True).start()
    try:
        input()
    finally:
        stop_event.set()
        pipeline.stop()


if __name__ == "__main__":
    main()
