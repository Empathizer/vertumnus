"""
Records a short voice sample for training-pipeline testing purposes.

Run interactively (not backgrounded) so you can see any macOS microphone
permission prompt and know exactly when to start talking:

    python record_voice_sample.py
"""
import sounddevice as sd
import soundfile as sf
import numpy as np

from audio.devices import print_devices, list_devices

OUT_PATH = "../training_data/my_voice.wav"
SR = 48000
DURATION = 90


def main() -> None:
    print_devices()
    valid = {d.index for d in list_devices()}
    while True:
        raw = input("Select INPUT device index (your mic): ").strip()
        try:
            device = int(raw)
        except ValueError:
            continue
        if device in valid:
            break
        print("Not a valid device index.")

    input(f"\nPress Enter, then start talking immediately. Recording runs for "
          f"{DURATION}s (keep talking naturally the whole time — read anything, "
          f"describe your day, whatever). Press Enter now to begin...")

    print("RECORDING NOW — talk!")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32", device=device)
    for remaining in range(DURATION, 0, -5):
        sd.sleep(5000)
        print(f"{remaining - 5}s left..." if remaining > 5 else "done.")
    sd.wait()

    sf.write(OUT_PATH, audio, SR)
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    print(f"\nSaved to {OUT_PATH}")
    print(f"RMS: {rms:.5f}  Peak: {peak:.5f}")
    if rms < 0.01:
        print("WARNING: this looks like silence/room noise, not speech. "
              "Check mic permissions (System Settings > Privacy & Security > "
              "Microphone) and try again.")
    else:
        print("Looks like real speech was captured.")


if __name__ == "__main__":
    main()
