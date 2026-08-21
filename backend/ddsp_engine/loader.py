"""Loads a DDSP-SVC rectified-flow checkpoint (model_N.pt + sibling config.yaml)
into runnable Unit2Wav + Vocoder objects.

Checkpoint layout (produced by ddsp_training/train_reflow.py, copied by
training.py into backend/models/{name}/):
  backend/models/{name}/model.pt     -- ckpt["model"] is the Unit2Wav state_dict
  backend/models/{name}/config.yaml  -- same schema ddsp_training/configs/reflow.yaml uses

Cannot call the vendored ddsp_training/reflow/vocoder.py's own
load_model_vocoder() unmodified: it builds Vocoder/Unit2Wav directly from the
relative pretrain paths in config.yaml, before there's any chance to rewrite
them to be cwd-independent. So this reimplements that ~20-line sequence
locally, inserting the path rewrite in between the YAML parse and the
Vocoder()/Unit2Wav() construction.
"""
import sys
from pathlib import Path

import torch
import yaml

DDSP_TRAINING_DIR = Path(__file__).resolve().parent.parent.parent / "ddsp_training"
if str(DDSP_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(DDSP_TRAINING_DIR))


class ModelLoadError(Exception):
    pass


def _load_config(config_path: Path):
    from logger.utils import DotDict

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ModelLoadError(f"Missing config.yaml next to checkpoint: {config_path}") from e
    except yaml.YAMLError as e:
        raise ModelLoadError(f"Malformed config.yaml at {config_path}: {e}") from e

    args = DotDict(raw)

    # DotDict.__getattr__ returns a COPY of nested dicts (DotDict(val)
    # shallow-copies), not a live reference -- `args.vocoder.ckpt = ...`
    # would silently mutate a throwaway copy and do nothing. Item-access
    # returns the real nested dict object, so mutate through that instead.
    try:
        args["vocoder"]["ckpt"] = str(DDSP_TRAINING_DIR / args["vocoder"]["ckpt"])
        args["data"]["encoder_ckpt"] = str(DDSP_TRAINING_DIR / args["data"]["encoder_ckpt"])
    except KeyError as e:
        raise ModelLoadError(f"config.yaml at {config_path} is missing required key: {e}") from e

    return args


def get_reflow_model(model_path: str, device: torch.device):
    model_path = Path(model_path)
    config_path = model_path.parent / "config.yaml"
    args = _load_config(config_path)

    if args.get("model", {}).get("type") != "RectifiedFlow":
        raise ModelLoadError(
            f"'{model_path}' is not a RectifiedFlow checkpoint "
            f"(config.yaml model.type={args.get('model', {}).get('type')!r}). "
            f"Only the reflow variant of DDSP-SVC is supported."
        )

    from reflow.vocoder import Unit2Wav, Vocoder

    try:
        vocoder = Vocoder(args["vocoder"]["type"], args["vocoder"]["ckpt"], device=device)
    except FileNotFoundError as e:
        raise ModelLoadError(f"NSF-HiFiGAN vocoder checkpoint not found: {e}") from e

    model = Unit2Wav(
        args["data"]["sampling_rate"],
        args["data"]["block_size"],
        args["model"]["win_length"],
        args["data"]["encoder_out_channels"],
        args["model"]["n_spk"],
        args["model"]["use_norm"],
        args["model"]["use_attention"],
        args["model"]["use_pitch_aug"],
        vocoder.dimension,
        args["model"]["n_aux_layers"],
        args["model"]["n_aux_chans"],
        args["model"]["n_layers"],
        args["model"]["n_chans"],
    )

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        raise ModelLoadError(f"Voice model checkpoint not found: {model_path}") from e
    except Exception as e:
        raise ModelLoadError(f"Failed to read voice model '{model_path}': {e}") from e

    if "model" not in ckpt:
        raise ModelLoadError(
            f"'{model_path}' does not look like a valid DDSP-SVC reflow checkpoint "
            f"(missing 'model' key)."
        )

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise ModelLoadError(
            f"Voice model '{model_path}' weights don't match the RectifiedFlow "
            f"architecture described by its config.yaml: {e}"
        ) from e

    model.eval().to(device)

    return model, vocoder, args
