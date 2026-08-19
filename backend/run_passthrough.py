"""
Stage 1 manual test: mic -> virtual mic pass-through.

Run: python run_passthrough.py
Lists devices, lets you pick input/output (virtual mic)/optional monitor,
then streams until you press Enter.
"""
from audio.devices import print_devices, list_devices
from audio.passthrough import PassthroughEngine


def prompt_index(label: str, allow_skip: bool = False) -> int | None:
    while True:
        raw = input(f"{label}: ").strip()
        if allow_skip and raw == "":
            return None
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a device index number.")
            continue
        valid_indices = {d.index for d in list_devices()}
        if idx not in valid_indices:
            print("Not a valid device index, try again.")
            continue
        return idx


def main() -> None:
    print_devices()

    input_device = prompt_index("Select INPUT device index (your mic)")
    output_device = prompt_index("Select OUTPUT device index (virtual mic, e.g. VB-Cable/BlackHole)")

    monitor_choice = input("Enable monitor (hear yourself)? [y/N]: ").strip().lower()
    monitor_device = None
    if monitor_choice == "y":
        monitor_device = prompt_index("Select MONITOR output device index (e.g. your speakers/headphones)")

    engine = PassthroughEngine(
        input_device=input_device,
        output_device=output_device,
        monitor_device=monitor_device,
    )

    print("\nStarting pass-through. Speak into your mic and check the output "
          "device in another app (or your monitor device).")
    print("Press Enter to stop.\n")

    engine.start()
    try:
        input()
    finally:
        engine.stop()
        print(f"Stopped. Output underruns: {engine.underrun_count}")


if __name__ == "__main__":
    main()
