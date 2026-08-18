"""Loads a user-supplied RVC .pth checkpoint into the matching Synthesizer class.

Checkpoint format (standard across the RVC ecosystem):
  cpt = {
    "weight": <state_dict>,
    "config": [...],       # positional args for the Synthesizer constructor
    "f0": 0 or 1,          # whether the model is pitch-conditioned
    "version": "v1"/"v2",  # 256-dim (v1) or 768-dim (v2) content features
  }
"""
import torch

from rvc.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

_ARCH_BY_VERSION_F0 = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
}


class ModelLoadError(Exception):
    pass


def get_synthesizer(pth_path: str, device: torch.device):
    try:
        cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        raise ModelLoadError(f"Voice model file not found: {pth_path}") from e
    except Exception as e:
        raise ModelLoadError(f"Failed to read voice model '{pth_path}': {e}") from e

    for key in ("weight", "config"):
        if key not in cpt:
            raise ModelLoadError(
                f"'{pth_path}' does not look like a valid RVC checkpoint "
                f"(missing '{key}'). Expected a .pth exported by RVC training/WebUI."
            )

    # n_spk must match the actual embedding table shape in the checkpoint.
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    arch = _ARCH_BY_VERSION_F0.get((version, if_f0))
    if arch is None:
        raise ModelLoadError(
            f"Unsupported RVC checkpoint version/f0 combination: "
            f"version={version!r}, f0={if_f0!r}"
        )

    net_g = arch(*cpt["config"], is_half=False)
    missing, unexpected = net_g.load_state_dict(cpt["weight"], strict=False)
    # Inference-only checkpoints never include enc_q.* (the training-only
    # posterior encoder, deleted below) — that's expected, not a mismatch.
    real_missing = [k for k in missing if not k.startswith("enc_q.")]
    if real_missing:
        raise ModelLoadError(
            f"Voice model '{pth_path}' is missing expected weights "
            f"(architecture mismatch?): {real_missing[:5]}..."
        )

    net_g = net_g.float()
    net_g.eval().to(device)
    # remove_weight_norm() on enc_q must happen before we delete it below.
    net_g.remove_weight_norm()
    del net_g.enc_q  # inference-only: drop the training-time posterior encoder

    return net_g, cpt, if_f0, version
