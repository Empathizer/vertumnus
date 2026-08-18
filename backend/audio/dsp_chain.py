"""
Stage 2: live-adjustable pedalboard effects chain.

Plugin instances are built once and mutated in place when sliders change
(instead of rebuilding the Pedalboard every chunk).

pedalboard's own streaming mode (reset=False) turned out not to work at all
for PitchShift at real-time chunk sizes — confirmed empirically it stayed
completely silent across a full second of continuous input. Each chunk is
instead processed independently with reset=True, given real preceding
audio as look-back context (instead of a cold start), and the boundary
between consecutive chunks' output is smoothed with a short raised-cosine
crossfade — the same overlap-add principle used for RVC's SOLA streaming
in rvc/streaming.py. This costs one crossfade-length of extra output
latency (~10ms by default) but removes the audible click/glitch that a
hard cut between independently-processed chunks otherwise produces.

Note on "Timbre": true formant shifting needs LPC/PSOLA resynthesis, which
is out of scope for a pedalboard filter chain. This is a simple spectral
tilt (a peak filter around the vocal formant region) — a lightweight
tone-shaping control, not real formant resynthesis. Once Stage 3's RVC
model is active, actual timbre change comes from the voice model itself.
"""
import threading
from dataclasses import dataclass

import numpy as np
import pedalboard


@dataclass
class EffectParams:
    pitch_semitones: float = 0.0       # -12..+12
    timbre_db: float = 0.0             # -12..+12, peak filter around 2.5kHz
    softness_hz: float = 20000.0       # lowpass cutoff; lower = softer/darker
    sharpness_db: float = 0.0          # 0..+12, high-shelf presence boost
    volume_db: float = 0.0             # -24..+12
    mix: float = 1.0                   # 0 = fully dry, 1 = fully wet


class MicNoiseGate:
    """Decides how much to attenuate background noise, based on the raw mic
    signal — but does NOT gate what feeds the RVC model's context window.

    Earlier version applied the gate directly to the audio fed into RVC
    conversion. That broke voice quality: RVC's real-time converter needs a
    continuous rolling context window to track pitch/voice characteristics
    smoothly, and zeroing the signal during every natural pause between
    words resets that context — so each word right after a pause started
    "cold" again, and only the tail of a longer continuous utterance (after
    the window re-filled with real audio) sounded right. Confirmed
    empirically: "first words garbled/quiet, last word in correct tune."

    Fix: compute the gate's gain envelope from the raw input (via
    compute_gain), but apply it only to the *final* output (post-conversion,
    post-effects) — see AudioPipeline._worker_loop. RVC always sees
    continuous, ungated audio; background noise still gets suppressed in
    what actually reaches the speaker/virtual mic.

    Custom implementation instead of pedalboard.NoiseGate: confirmed
    empirically that pedalboard's gate has no hysteresis or hold time, so
    background noise hovering anywhere near the threshold makes it flicker
    open/closed dozens of times a second — average energy barely drops, and
    each flicker is an audible click, which sounds like a buzz/rattle
    during "silence" rather than actual quiet. This gate uses a HIGHER
    threshold to open than to close (hysteresis, so borderline noise can't
    make it flicker) plus a minimum hold time after the signal drops before
    it's allowed to close at all, then ramps gain smoothly (no per-sample
    discontinuity) rather than switching instantly.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold_db: float = -40.0,
        hysteresis_db: float = 6.0,
        hold_ms: float = 150.0,
        attack_ms: float = 12.0,
        release_ms: float = 200.0,
    ):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self.threshold_db = threshold_db
        self._hysteresis_db = hysteresis_db
        self._hold_samples = int(sample_rate * hold_ms / 1000)
        self._attack_ms = attack_ms
        self._release_ms = release_ms

        self._gain = 0.0          # current smoothed gain, 0..1
        self._is_open = False
        self._silence_samples = 0  # consecutive samples spent below the close threshold

    def set_threshold(self, threshold_db: float) -> None:
        with self._lock:
            self.threshold_db = threshold_db

    def compute_gain(self, audio: np.ndarray) -> np.ndarray:
        """Returns a per-sample gain ramp (length == len(audio)) based on
        this chunk's input level. Does not touch the audio itself — apply
        the returned ramp to whatever signal you actually want gated
        (typically the final output, not RVC's input — see class docstring)."""
        n = len(audio)
        with self._lock:
            open_db = self.threshold_db
            close_db = self.threshold_db - self._hysteresis_db

            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
            level_db = 20.0 * np.log10(rms)

            if level_db > open_db:
                self._is_open = True
                self._silence_samples = 0
            elif level_db < close_db:
                self._silence_samples += n
                if self._silence_samples >= self._hold_samples:
                    self._is_open = False
            # else: in the hysteresis dead-band — hold current open/close state

            target_gain = 1.0 if self._is_open else 0.0
            ramp_ms = self._attack_ms if target_gain > self._gain else self._release_ms
            max_delta = n / (self.sample_rate * ramp_ms / 1000) if ramp_ms > 0 else 1.0

            start_gain = self._gain
            end_gain = start_gain + np.clip(target_gain - start_gain, -max_delta, max_delta)
            gain_ramp = np.linspace(start_gain, end_gain, n, dtype=np.float32)
            self._gain = float(end_gain)

        return gain_ramp

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Convenience: computes and applies the gain in one step. Only use
        this where gating the input itself is actually fine (e.g. DSP-only
        passthrough with no RVC model loaded) — NOT ahead of RVC conversion."""
        return audio * self.compute_gain(audio)


class EffectChain:
    def __init__(self, sample_rate: int, crossfade_seconds: float = 0.01):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self.params = EffectParams()

        self._pitch = pedalboard.PitchShift(semitones=0.0)
        self._timbre = pedalboard.PeakFilter(cutoff_frequency_hz=2500.0, gain_db=0.0, q=0.7)
        self._softness = pedalboard.LowpassFilter(cutoff_frequency_hz=20000.0)
        self._sharpness = pedalboard.HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=0.0)
        self._volume = pedalboard.Gain(gain_db=0.0)

        self._board = pedalboard.Pedalboard(
            [self._pitch, self._timbre, self._softness, self._sharpness, self._volume]
        )

        self._overlap = max(32, int(sample_rate * crossfade_seconds))
        fade_in = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, self._overlap, dtype=np.float32)) ** 2
        self._fade_in = fade_in.astype(np.float32)
        self._fade_out = 1.0 - self._fade_in

        self._history = np.zeros(self._overlap, dtype=np.float32)
        self._pending_tail: np.ndarray | None = None

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self.params, key):
                    raise ValueError(f"Unknown effect parameter: {key}")
                setattr(self.params, key, value)

            self._pitch.semitones = self.params.pitch_semitones
            self._timbre.gain_db = self.params.timbre_db
            self._softness.cutoff_frequency_hz = self.params.softness_hz
            self._sharpness.gain_db = self.params.sharpness_db
            self._volume.gain_db = self.params.volume_db

    def process(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        overlap = self._overlap
        with self._lock:
            mix = self.params.mix
            if mix <= 0.0:
                return audio

            if n <= overlap:
                # Chunk too short for the crossfade scheme; fall back to a
                # plain reset=True call (may click, but won't crash).
                wet = self._board(audio, self.sample_rate)
                return wet if mix >= 1.0 else audio * (1.0 - mix) + wet * mix

            window = np.concatenate([self._history, audio])
            out = self._board(window, self.sample_rate)  # reset=True (default)
            head = out[:overlap]
            new_part = out[overlap:]

            if self._pending_tail is None:
                wet = np.concatenate([np.zeros(overlap, dtype=np.float32), new_part[: n - overlap]])
            else:
                crossfaded = self._pending_tail * self._fade_out + head * self._fade_in
                wet = np.concatenate([crossfaded, new_part[: n - overlap]])

            self._pending_tail = new_part[-overlap:]
            self._history = audio[-overlap:]

        if mix >= 1.0:
            return wet
        return audio * (1.0 - mix) + wet * mix
