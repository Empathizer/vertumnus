"""Discovers user-supplied RVC voice models dropped into backend/models/.

No models are bundled with this app — only user's own/licensed .pth files,
optionally paired with a same-stem .index retrieval file.
"""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class VoiceModelInfo:
    name: str
    pth_path: str
    index_path: str | None


def list_voice_models() -> list[VoiceModelInfo]:
    MODELS_DIR.mkdir(exist_ok=True)
    models = []
    for pth in sorted(MODELS_DIR.glob("*.pth")):
        index = MODELS_DIR / f"{pth.stem}.index"
        models.append(
            VoiceModelInfo(
                name=pth.stem,
                pth_path=str(pth),
                index_path=str(index) if index.is_file() else None,
            )
        )
    return models
