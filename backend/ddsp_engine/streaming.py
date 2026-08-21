"""
Turns DDSPVoice.infer_chunk() (single-window conversion) into continuous
real-time streaming: keeps a sliding window of recent audio for model
context, and uses SOLA (synchronized overlap-add) to blend consecutive
inferred windows together without clicks at the seams.

Unlike rvc/streaming.py, DDSP-SVC's reference implementation (gui_reflow.py)
reinfers the ENTIRE rolling window on every callback rather than carrying an
incremental pitch/feature cache forward between calls — there's no
cache_pitch/cache_pitchf equivalent here, since rectified-flow conditioning
uses continuous f0 Hz values recomputed fresh each call, not a bucketed
cache. Measured steady-state cost for this (RTF~0.06 after one-time CUDA
warm-up, absorbed in DDSPVoice.__init__) comfortably fits the block/extra
time budget below, but this is the single biggest latency-tuning unknown
in the whole DDSP integration — see the plan's Section 5 if these defaults
need adjusting once real end-to-end latency is measured.

`silence_front` tells the model to skip DDSP-resynthesizing the already-
processed leading context of the window (gui_reflow.py's own optimization).
This makes the model's returned audio shorter than the input window, so —
same as gui_reflow.py's own audio_callback and RVC's process_block — this
always slices the *tail* of whatever infer_chunk() returns, never assumes
an exact absolute length.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from ddsp_engine.engine import DDSPVoice


class DDSPRealtimeConverter:
    def __init__(
        self,
        voice: DDSPVoice,
        device: torch.device,
        block_time: float = 0.5,
        crossfade_time: float = 0.04,
        extra_time: float = 1.0,
        f0_up_key: int = 0,
        infer_step: int = 30,
        t_start: float = 0.7,
        sampling_method: str = "euler",
    ):
        self.voice = voice
        self.device = device
        self.f0_up_key = f0_up_key
        self.infer_step = infer_step
        self.t_start = t_start
        self.sampling_method = sampling_method

        self.target_sample_rate = voice.target_sample_rate
        hop = voice.hop_size

        self.block_frame = int(np.round(block_time * self.target_sample_rate / hop)) * hop
        self.crossfade_frame = int(np.round(crossfade_time * self.target_sample_rate / hop)) * hop
        self.sola_search_frame = int(np.round(0.01 * self.target_sample_rate / hop)) * hop or hop
        self.extra_frame = int(np.round(extra_time * self.target_sample_rate / hop)) * hop

        total_frame = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(total_frame, device=device, dtype=torch.float32)

        self.sola_buffer = torch.zeros(self.crossfade_frame, device=device, dtype=torch.float32)

        self.fade_in_window = (
            torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=device))
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        # Mirrors gui_reflow.py's own f_safe_prefix_pad_length formula:
        # skip resynthesizing everything except the crossfade lookback plus
        # a small safety margin.
        self.silence_front = max(0.0, extra_time - crossfade_time - 0.03)

        self.last_infer_ms = 0.0

        # Same rationale as rvc/streaming.py: neural vocoder output (both
        # RVC's and NSF-HiFiGAN here) runs quieter than natural speech —
        # auto gain-match to the actual input loudness per block instead of
        # a manually-tuned slider. Smoothed to avoid audible pumping.
        self._gain_smoothed = 1.0
        self._gain_smoothing = 0.3
        self._max_gain = 8.0

    def latency_seconds(self) -> float:
        """Extra look-ahead latency the SOLA algorithm adds on top of block_time."""
        return self.crossfade_frame / self.target_sample_rate

    def process_block(self, indata: np.ndarray) -> np.ndarray:
        """indata: mono float32 numpy array of exactly self.block_frame samples,
        at self.target_sample_rate. Returns a same-length converted chunk."""
        t0 = time.perf_counter()

        chunk = torch.from_numpy(indata).to(self.device)
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = chunk

        window = self.input_wav.unsqueeze(0)
        window_np = self.input_wav.cpu().numpy()

        infer_wav = self.voice.infer_chunk(
            window,
            window_np,
            f0_up_key=self.f0_up_key,
            infer_step=self.infer_step,
            t_start=self.t_start,
            sampling_method=self.sampling_method,
            silence_front=self.silence_front,
        )
        # infer_chunk's output is shorter than the input window whenever
        # silence_front > 0 (the model skips resynthesizing that prefix) --
        # always index from the end, never assume an exact absolute length,
        # same as gui_reflow.py's own audio_callback and RVC's process_block.
        infer_wav = infer_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :]

        # SOLA: find the offset in infer_wav that best matches the tail of
        # the previous chunk (self.sola_buffer), then crossfade the overlap.
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
