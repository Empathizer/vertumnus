"""
Core DDSP-SVC (rectified-flow) voice-conversion engine: one loaded voice
model + ContentVec content encoder + RMVPE pitch estimator + NSF-HiFiGAN
vocoder, wired together to convert one chunk of audio at a time.

This does the per-chunk math only. Sliding-window buffering and crossfading
for continuous real-time streaming lives in ddsp/streaming.py, which calls
DDSPVoice.infer_chunk() once per block — same split as rvc/engine.py vs.
rvc/streaming.py.

Unlike RVC, there's no retrieval-index concept here, and pitch is continuous
Hz values recomputed fresh every call rather than a bucketed cache the caller
carries forward.
"""
import os
from pathlib import Path

import numpy as np
import torch

from ddsp_engine.loader import ModelLoadError, get_reflow_model, DDSP_TRAINING_DIR

DDSP_HUBERT_SAMPLE_RATE = 16000  # ContentVec's own encoder_sample_rate; RMVPE also runs at 16k internally


class DDSPNotFoundError(Exception):
    pass


_units_encoder_cache: dict[str, object] = {}


def _get_or_build_units_encoder(encoder: str, encoder_ckpt: str, encoder_sample_rate: int,
                                 encoder_hop_size: int, device: torch.device):
    """ContentVec is expensive to construct (loads a HuggingFace HubertModel) —
    cache it once per (encoder type, checkpoint, device), same pattern as
    rvc/engine.py's load_hubert() caching."""
    from ddsp.vocoder import Units_Encoder

    key = f"{encoder}:{encoder_ckpt}:{device}"
    if key in _units_encoder_cache:
        return _units_encoder_cache[key]

    if not Path(encoder_ckpt).is_file():
        raise DDSPNotFoundError(
            f"ContentVec content encoder not found at {encoder_ckpt}. "
            f"Download it per the DDSP-SVC setup instructions."
        )

    encoder_obj = Units_Encoder(
        encoder, encoder_ckpt, encoder_sample_rate, encoder_hop_size, device=device,
    )
    _units_encoder_cache[key] = encoder_obj
    return encoder_obj


def _warm_rmvpe(sample_rate: int, hop_size: int):
    """RMVPE's checkpoint path is hardcoded relative ('pretrain/rmvpe/model.pt')
    inside ddsp_training/ddsp/vocoder.py, not sourced from config — it only
    resolves if cwd is ddsp_training/, which it isn't for this app's server
    process. RMVPE's model instance is cached module-globally by DDSP-SVC's
    own code (F0_KERNEL dict) keyed by extractor name, so this only needs to
    run once, here at model-load time — not per audio chunk. Scoped chdir
    mirrors the existing precedent in rvc/engine.py's load_hubert(), which
    temporarily monkeypatches torch.load and restores it in finally."""
    from ddsp.vocoder import F0_Extractor

    cwd = os.getcwd()
    os.chdir(DDSP_TRAINING_DIR)
    try:
        F0_Extractor("rmvpe", sample_rate, hop_size, f0_min=65, f0_max=800)
    except FileNotFoundError as e:
        raise DDSPNotFoundError(
            f"RMVPE pitch model not found at {DDSP_TRAINING_DIR / 'pretrain' / 'rmvpe' / 'model.pt'}. "
            f"Download it per the DDSP-SVC setup instructions."
        ) from e
    finally:
        os.chdir(cwd)


class DDSPVoice:
    """One loaded DDSP-SVC reflow checkpoint (model.pt + config.yaml), ready
    to convert audio chunks."""

    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model_path = model_path

        self.model, self.vocoder, self.args = get_reflow_model(model_path, device)
        self.target_sample_rate = self.args["data"]["sampling_rate"]
        self.hop_size = self.args["data"]["block_size"]

        self.units_encoder = _get_or_build_units_encoder(
            self.args["data"]["encoder"],
            self.args["data"]["encoder_ckpt"],
            self.args["data"]["encoder_sample_rate"],
            self.args["data"]["encoder_hop_size"],
            device,
        )

        _warm_rmvpe(self.target_sample_rate, self.hop_size)

        # Absorb the one-time CUDA kernel-compile/warm-up cost (measured
        # ~1-1.5s cold vs. ~0.06 RTF steady-state) here at load time, so it
        # doesn't stall the user's first real audio chunk.
        self._warm_up()

    def _warm_up(self):
        dummy = torch.zeros(1, self.target_sample_rate, device=self.device)
        dummy_np = dummy[0].cpu().numpy()
        with torch.no_grad():
            self.infer_chunk(dummy, dummy_np)

    # -- full chunk inference -----------------------------------------------

    def infer_chunk(
        self,
        audio: torch.Tensor,
        audio_np: np.ndarray,
        f0_up_key: int = 0,
        infer_step: int = 30,
        t_start: float = 0.7,
        sampling_method: str = "euler",
        silence_front: float = 0.0,
    ) -> torch.Tensor:
        """Converts one window of mono audio (whole rolling buffer, owned by
        ddsp/streaming.py) at self.target_sample_rate. No cross-call state
        kept here — every call is self-contained over whatever window
        streaming.py hands it, matching DDSP-SVC's own reference design
        (SvcDDSP.infer in gui_reflow.py) rather than RVC's incremental
        cache_pitch/cache_pitchf carry-forward.

        audio: torch.Tensor[1, T] float32 on self.device.
        audio_np: same audio as a 1D numpy array (avoids a redundant GPU->CPU
        copy inside the pitch/volume extractors, which need numpy).
        """
        from ddsp.vocoder import F0_Extractor, Volume_Extractor

        f0_extractor = F0_Extractor(
            "rmvpe", self.target_sample_rate, self.hop_size, f0_min=65, f0_max=800
        )
        volume_extractor = Volume_Extractor(self.hop_size)

        with torch.no_grad():
            units = self.units_encoder.encode(audio, self.target_sample_rate, self.hop_size)

            f0 = f0_extractor.extract(
                audio_np, uv_interp=True, device=self.device, silence_front=silence_front
            )
            f0 = f0 * pow(2, f0_up_key / 12)
            f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(-1).to(self.device)

            volume = volume_extractor.extract(audio_np)
            volume_t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(-1).to(self.device)

            out_wav = self.model(
                units, f0_t, volume_t,
                spk_id=torch.LongTensor([1]).to(self.device),
                spk_mix_dict=None,
                aug_shift=None,
                vocoder=self.vocoder,
                infer=True,
                return_wav=True,
                infer_step=infer_step,
                method=sampling_method,
                t_start=t_start,
                silence_front=silence_front,
                use_tqdm=False,
            )

        return out_wav[0, 0].data.float()
