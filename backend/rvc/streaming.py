"""
Turns RVCVoice.infer() (single-chunk conversion) into continuous real-time
streaming: keeps a sliding window of recent audio for model context, and
uses SOLA (synchronized overlap-add, from the DDSP-SVC project) to blend
consecutive inferred chunks together without clicks at the seams.

This is deliberately close to RVC-Project's own gui_v1.py real-time client,
minus its optional noise-reduction/RMS-mixing stages (left out for a first
cut — the pedalboard DSP chain already covers tone-shaping).
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from rvc.engine import RVCVoice

TARGET_SAMPLE_RATE_16K = 16000


class RealtimeConverter:
    def __init__(
        self,
        voice: RVCVoice,
        device: torch.device,
        block_time: float = 0.25,
        crossfade_time: float = 0.05,
        extra_time: float = 2.5,
        f0_up_key: int = 0,
        f0_method: str = "harvest",
    ):
        self.voice = voice
        self.device = device
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method

        self.target_sample_rate = voice.target_sample_rate
        zc = self.target_sample_rate // 100
        self.zc = zc

        self.block_frame = int(np.round(block_time * self.target_sample_rate / zc)) * zc
        self.block_frame_16k = 160 * self.block_frame // zc
        self.crossfade_frame = int(np.round(crossfade_time * self.target_sample_rate / zc)) * zc
        self.sola_search_frame = zc
        self.extra_frame = int(np.round(extra_time * self.target_sample_rate / zc)) * zc

        total_frame = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(total_frame, device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(160 * total_frame // zc, device=device, dtype=torch.float32)

        self.cache_pitch = np.zeros(total_frame // zc, dtype=np.int32)
        self.cache_pitchf = np.zeros(total_frame // zc, dtype=np.float64)

        self.sola_buffer = torch.zeros(self.crossfade_frame, device=device, dtype=torch.float32)
        self.valid_rate = 1 - (self.extra_frame - 1) / total_frame

        self.fade_in_window = (
            torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=device))
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler = tat.Resample(orig_freq=self.target_sample_rate, new_freq=TARGET_SAMPLE_RATE_16K).to(device)

        self.last_infer_ms = 0.0

        # RVC's raw vocoder output is inherently quieter than natural human
        # speech (confirmed empirically — measured well below input level
        # even for clearly-voiced blocks). Auto gain-match the converted
        # output's loudness to the actual input loudness per block, instead
        # of requiring a manually-tuned Volume slider that would need
        # re-guessing for every different voice model. Smoothed across
        # blocks so gain doesn't pump audibly; clamped so a near-silent
        # block (denominator near zero) can't produce a huge gain spike.
        self._gain_smoothed = 1.0
        self._gain_smoothing = 0.3  # higher = slower to react, less pumping
        self._max_gain = 8.0  # +18dB ceiling

    def latency_seconds(self) -> float:
        """Extra look-ahead latency the SOLA algorithm adds on top of block_time."""
        return self.crossfade_frame / self.target_sample_rate

    def process_block(self, indata: np.ndarray) -> np.ndarray:
        """indata: mono float32 numpy array of exactly self.block_frame samples,
        at self.target_sample_rate. Returns a same-length converted chunk."""
        import time

        t0 = time.perf_counter()

        chunk = torch.from_numpy(indata).to(self.device)
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = chunk

        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
            self.input_wav[-self.block_frame - 2 * self.zc :]
        )[160:]

        f0_extractor_frame = self.block_frame_16k + 800
        f0_source = self.input_wav_res[-f0_extractor_frame:].cpu().numpy()

        infer_wav = self.voice.infer(
            self.input_wav_res,
            f0_source,
            self.block_frame_16k,
            self.valid_rate,
            self.cache_pitch,
            self.cache_pitchf,
            self.f0_up_key,
            self.f0_method,
        )
        infer_wav = infer_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :]

        # SOLA: find the offset in infer_wav that best matches the tail of the
        # previous chunk (self.sola_buffer), then crossfade the overlap.
        conv_input = infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input**2, torch.ones(1, 1, self.crossfade_frame, device=self.device)) + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()

        infer_wav = infer_wav[sola_offset : sola_offset + self.block_frame + self.crossfade_frame]
        infer_wav[: self.crossfade_frame] *= self.fade_in_window
        infer_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[-self.crossfade_frame :]

        out = infer_wav[: -self.crossfade_frame]

        dry_ref = self.input_wav[-self.block_frame :]
        rms_in = torch.sqrt(torch.mean(dry_ref**2) + 1e-8)
        rms_out = torch.sqrt(torch.mean(out**2) + 1e-8)
        target_gain = torch.clamp(rms_in / rms_out, max=self._max_gain).item()
        self._gain_smoothed = (
            self._gain_smoothing * self._gain_smoothed + (1 - self._gain_smoothing) * target_gain
        )
        out = out * self._gain_smoothed

        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        return out.cpu().numpy()
