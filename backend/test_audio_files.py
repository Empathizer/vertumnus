"""Discovers WAV files dropped into backend/test_audio/, for the "use a file
instead of the mic" input mode — useful for controlled, repeatable testing
without live mic background noise/feedback in the way.
"""
from pathlib import Path

TEST_AUDIO_DIR = Path(__file__).resolve().parent / "test_audio"


def list_test_audio_files() -> list[str]:
    TEST_AUDIO_DIR.mkdir(exist_ok=True)
    return sorted(str(p) for p in TEST_AUDIO_DIR.glob("*.wav"))
