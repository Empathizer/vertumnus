"""Discovers user-supplied voice models dropped into backend/models/.

Two on-disk formats are supported, sharing this one namespace:
  - RVC: a flat {name}.pth (+ optional same-stem {name}.index retrieval file).
  - DDSP-SVC (rectified-flow): a subdirectory {name}/ containing model.pt +
    config.yaml (a checkpoint needs both, and the loader assumes they're
    siblings -- see backend/ddsp_engine/loader.py).

No models are bundled with this app -- only user's own/licensed/trained files.
"""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class VoiceModelInfo:
    name: str
    engine: str  # "rvc" | "ddsp"
    pth_path: str  # RVC: the .pth. DDSP: the model.pt. Kept as one field name
                    # across both formats so callers (ws_server.py, the
                    # frontend) can treat it as an opaque "primary model
                    # file" identifier without branching on engine.
    index_path: str | None  # RVC only; always None for DDSP.
    config_path: str | None  # DDSP only; always None for RVC.


def list_voice_models() -> list[VoiceModelInfo]:
    MODELS_DIR.mkdir(exist_ok=True)
    models = []
    for pth in sorted(MODELS_DIR.glob("*.pth")):
        index = MODELS_DIR / f"{pth.stem}.index"
        models.append(
            VoiceModelInfo(
                name=pth.stem,
                engine="rvc",
                pth_path=str(pth),
                index_path=str(index) if index.is_file() else None,
                config_path=None,
            )
        )
    for d in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        model_pt = d / "model.pt"
        config_yaml = d / "config.yaml"
        if model_pt.is_file() and config_yaml.is_file():
            models.append(
                VoiceModelInfo(
                    name=d.name,
                    engine="ddsp",
                    pth_path=str(model_pt),
                    index_path=None,
                    config_path=str(config_yaml),
                )
            )
    return models
