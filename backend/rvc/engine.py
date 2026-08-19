"""
Core RVC voice-conversion engine: one loaded voice model + optional retrieval
index + pitch estimator + HuBERT content encoder, wired together to convert
one chunk of 16kHz mono audio at a time.

This does the per-chunk math only. Sliding-window buffering and crossfading
for continuous real-time streaming lives in rvc/streaming.py, which calls
RVCVoice.infer() once per block. Kept separate so this class can also be
used for simple one-shot inference/testing.
"""
import logging
import traceback
from pathlib import Path

import faiss
import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F

# torch and faiss each bundle their own OpenMP runtime; once both have done
# real threaded work in the same process, the resulting thread-pool
# collision reliably segfaults faiss's IVF search (confirmed empirically —
# random data through the same index never crashed, but real HuBERT-derived
# features did, consistently, on macOS/arm64). Forcing faiss to a single
# thread avoids the collision. Index search is not the bottleneck here, so
# this costs negligible latency.
faiss.omp_set_num_threads(1)

from rvc.loader import ModelLoadError, get_synthesizer

logger = logging.getLogger(__name__)

HUBERT_SAMPLE_RATE = 16000
DEFAULT_HUBERT_PATH = Path(__file__).resolve().parent.parent / "models" / "hubert_base.pt"


class HubertNotFoundError(Exception):
    pass


_hubert_cache: dict[str, torch.nn.Module] = {}


def load_hubert(device: torch.device, hubert_path: Path = DEFAULT_HUBERT_PATH):
    """Loads (and caches) the fairseq HuBERT content encoder used by all RVC models.

    This is the generic speech feature extractor RVC's architecture always
    requires — not a voice model. It is not bundled with this app; download
    it once per README instructions.
    """
    key = f"{hubert_path}:{device}"
    if key in _hubert_cache:
        return _hubert_cache[key]

    if not hubert_path.is_file():
        raise HubertNotFoundError(
            f"HuBERT content encoder not found at {hubert_path}. "
            f"Download hubert_base.pt per the README and place it there."
        )

    import fairseq

    # fairseq's checkpoint_utils calls torch.load() without weights_only=False,
    # which crashes under PyTorch>=2.6's new weights_only=True default (it
    # can't unpickle fairseq.data.dictionary.Dictionary). Patch torch.load
    # for the duration of this call rather than editing fairseq's installed
    # source, so this keeps working across fresh installs/platforms.
    _original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(hubert_path)], suffix=""
        )
    finally:
        torch.load = _original_torch_load
    model = models[0].to(device).float().eval()
    _hubert_cache[key] = model
    return model


class RVCVoice:
    """One loaded .pth (+ optional .index), ready to convert audio chunks."""

    def __init__(
        self,
        pth_path: str,
        device: torch.device,
        index_path: str | None = None,
        index_rate: float = 0.75,
        hubert_path: Path = DEFAULT_HUBERT_PATH,
    ):
        self.device = device
        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate if index_path else 0.0

        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.hubert_model = load_hubert(device, hubert_path)

        self.net_g, cpt, self.if_f0, self.version = get_synthesizer(pth_path, device)
        self.target_sample_rate = cpt["config"][-1]

        self.index = None
        self.index_vectors = None
        if index_path:
            try:
                self.index = faiss.read_index(index_path)
                self.index_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception as e:
                raise ModelLoadError(f"Failed to load retrieval index '{index_path}': {e}") from e

    # -- pitch extraction -------------------------------------------------

    def _f0_post(self, f0: np.ndarray):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0

    def get_f0(self, audio_16k: np.ndarray, f0_up_key: int, method: str = "harvest"):
        """method: 'harvest' (pyworld, CPU, robust) | 'pm' (parselmouth, fast) |
        'crepe' (torchcrepe, GPU, best quality)."""
        if method == "crepe":
            import torchcrepe

            audio_t = torch.tensor(np.copy(audio_16k))[None].float()
            f0, periodicity = torchcrepe.predict(
                audio_t,
                HUBERT_SAMPLE_RATE,
                160,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=512,
                device=self.device,
                return_periodicity=True,
            )
            periodicity = torchcrepe.filter.median(periodicity, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[periodicity < 0.1] = 0
            f0 = f0[0].cpu().numpy()
            f0 *= pow(2, f0_up_key / 12)
            return self._f0_post(f0)

        if method == "pm":
            p_len = audio_16k.shape[0] // 160 + 1
            f0 = (
                parselmouth.Sound(audio_16k, HUBERT_SAMPLE_RATE)
                .to_pitch_ac(
                    time_step=0.01,
                    voicing_threshold=0.6,
                    pitch_floor=50,
                    pitch_ceiling=1100,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
            f0 *= pow(2, f0_up_key / 12)
            return self._f0_post(f0)

        # harvest (default): robust CPU pitch tracker
        f0, _ = pyworld.harvest(
            audio_16k.astype(np.double),
            fs=HUBERT_SAMPLE_RATE,
            f0_ceil=1100,
            f0_floor=50,
            frame_period=10,
        )
        f0 = signal.medfilt(f0, 3)
        f0 *= pow(2, f0_up_key / 12)
        return self._f0_post(f0)

    # -- content features ---------------------------------------------------

    def extract_content_features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        feats = audio_16k.view(1, -1).float().to(self.device)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        with torch.no_grad():
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.hubert_model.extract_features(**inputs)
            feats = (
                self.hubert_model.final_proj(logits[0])
                if self.version == "v1"
                else logits[0]
            )
        return F.pad(feats, (0, 0, 1, 0))

    def apply_index_retrieval(self, feats: torch.Tensor, rate: float) -> torch.Tensor:
        if self.index is None or self.index_rate == 0:
            return feats
        try:
            leng_replace_head = int(rate * feats[0].shape[0])
            npy = feats[0][-leng_replace_head:].cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats[0][-leng_replace_head:] = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                + (1 - self.index_rate) * feats[0][-leng_replace_head:]
            )
        except Exception:
            logger.warning("Index retrieval failed:\n%s", traceback.format_exc())
        return feats

    # -- full chunk inference -----------------------------------------------

    def infer(
        self,
        audio_16k: torch.Tensor,
        f0_source_audio_16k: np.ndarray,
        block_frame_16k: int,
        rate: float,
        cache_pitch: np.ndarray | None,
        cache_pitchf: np.ndarray | None,
        f0_up_key: int = 0,
        f0_method: str = "harvest",
    ) -> torch.Tensor:
        """Converts one chunk. cache_pitch/cache_pitchf are rolling buffers the
        caller owns across calls (see rvc/streaming.py)."""
        feats = self.extract_content_features(audio_16k)
        feats = self.apply_index_retrieval(feats, rate)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(f0_source_audio_16k, f0_up_key, f0_method)
            start_frame = block_frame_16k // 160
            end_frame = len(cache_pitch) - (pitch.shape[0] - 4) + start_frame
            cache_pitch[:] = np.append(cache_pitch[start_frame:end_frame], pitch[3:-1])
            cache_pitchf[:] = np.append(cache_pitchf[start_frame:end_frame], pitchf[3:-1])
            p_len = min(feats.shape[1], 13000, cache_pitch.shape[0])
        else:
            p_len = min(feats.shape[1], 13000)

        feats = feats[:, :p_len, :]
        sid = torch.LongTensor([0]).to(self.device)
        p_len_t = torch.LongTensor([p_len]).to(self.device)

        with torch.no_grad():
            if self.if_f0 == 1:
                pitch_t = torch.LongTensor(cache_pitch[:p_len]).unsqueeze(0).to(self.device)
                pitchf_t = torch.FloatTensor(cache_pitchf[:p_len]).unsqueeze(0).to(self.device)
                out = self.net_g.infer(
                    feats, p_len_t, pitch_t, pitchf_t, sid, torch.FloatTensor([rate])
                )[0][0, 0].data.float()
            else:
                out = self.net_g.infer(
                    feats, p_len_t, sid, torch.FloatTensor([rate])
                )[0][0, 0].data.float()

        return out
