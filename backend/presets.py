"""User effect/voice presets, saved as local JSON files (not browser storage)."""
import json
from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _safe_path(name: str) -> Path:
    if not name or any(c in name for c in "/\\.."):
        raise ValueError(f"Invalid preset name: {name!r}")
    return PRESETS_DIR / f"{name}.json"


def list_presets() -> list[str]:
    PRESETS_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def save_preset(name: str, data: dict) -> None:
    PRESETS_DIR.mkdir(exist_ok=True)
    path = _safe_path(name)
    path.write_text(json.dumps(data, indent=2))


def load_preset(name: str) -> dict:
    path = _safe_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"No such preset: {name}")
    return json.loads(path.read_text())


def delete_preset(name: str) -> None:
    path = _safe_path(name)
    if path.is_file():
        path.unlink()
