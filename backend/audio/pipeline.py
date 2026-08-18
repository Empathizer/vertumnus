"""
Stage 3: the real audio engine used by the app.

Architecture (per the hard constraint that the audio callback must never
block on ML inference):

  mic --[InputStream callback: just copies into a queue]--> in_queue
                                                                 |
                                                    worker thread (this file)
                                                    RVC conversion (if a voice
                                                    is loaded) -> DSP chain
                                                                 |
                                                              out_queue(s)
                                                                 |
  virtual mic <--[OutputStream callback: just pops the queue]--+--> monitor
"""
import queue
import threading
import time
from dataclasses import dataclass

import librosa
import numpy as np
import sounddevice as sd
import torch

from audio.dsp_chain import EffectChain, MicNoiseGate
from rvc.engine import HubertNotFoundError, RVCVoice
from rvc.loader import ModelLoadError
from rvc.streaming import RealtimeConverter

QUEUE_MAXSIZE = 16
DEFAULT_SAMPLE_RATE = 48000
# Used only when no voice model is loaded yet (RVC's own block_time governs
# block size once a voice is active). 480 (10ms) is too tight a deadline for
# Python-level per-chunk processing under normal OS scheduling and caused
# real underruns; 2048 (~43ms) is still low-latency but far more robust.
DEFAULT_BLOCKSIZE = 2048

VIRTUAL_MIC_NAME_HINTS = ("CABLE Input", "VB-Audio", "BlackHole")


class OutputResampleBuffer:
    """Feeds a fixed-blocksize OutputStream callback from audio produced at a
    different sample rate than the device actually runs at.

    Some virtual devices (BlackHole in particular) don't do real-time sample
    rate conversion like real hardware does — they run at one fixed nominal
    rate (set via Audio MIDI Setup), and playing audio generated at a
    different rate through them plays back at the wrong speed/pitch
    (confirmed empirically: a 40kHz-generated stream through BlackHole
    pinned to 44100Hz played back audibly faster). Resampling explicitly to
    the device's actual rate here fixes that regardless of what nominal
    rate the device happens to be configured for.

    Pushed chunks (variable length after resampling) accumulate in a FIFO
    buffer; the output callback pops exactly the fixed frame count it needs
    per call, since PortAudio callbacks require a constant blocksize.
    """

    def __init__(self, in_sr: int, out_sr: int, max_buffer_seconds: float = 1.0):
        self.in_sr = in_sr
        self.out_sr = out_sr
        self._needs_resample = in_sr != out_sr
        self._max_buffer_samples = int(out_sr * max_buffer_seconds)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        if self._needs_resample:
            chunk = librosa.resample(chunk, orig_sr=self.in_sr, target_sr=self.out_sr).astype(np.float32)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            if len(self._buffer) > self._max_buffer_samples:
                # Device consuming slower than we're producing — drop the
                # oldest excess rather than let latency grow unbounded.
                self._buffer = self._buffer[-self._max_buffer_samples :]

    def pop(self, n: int) -> tuple[np.ndarray, bool]:
        """Returns (chunk_of_length_n, had_enough_data)."""
        with self._lock:
            if len(self._buffer) >= n:
                out = self._buffer[:n]
                self._buffer = self._buffer[n:]
                return out, True
            out = np.zeros(n, dtype=np.float32)
            out[: len(self._buffer)] = self._buffer
            self._buffer = np.zeros(0, dtype=np.float32)
            return out, False


class PipelineError(Exception):
    """kind is one of: model_not_found, device_not_found, cuda_unavailable,
    cuda_kernel_mismatch, virtual_mic_not_installed, unknown."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def check_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise PipelineError(
            "cuda_unavailable",
            "CUDA is not available to PyTorch. Confirm you installed the cu128 "
            "wheel (never plain 'pip install torch') and that your NVIDIA driver "
            "is current (580.82+). See check_gpu.py.",
        )
    try:
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        (a @ b).sum().item()
    except RuntimeError as e:
        raise PipelineError(
            "cuda_kernel_mismatch",
            f"CUDA matmul failed on {device}: {e}\n"
            "This is the sm_120 kernel mismatch (RTX 5060 Ti/Blackwell). Switch to "
            "the nightly cu128 torch line — see setup.bat.",
        ) from e


def find_virtual_mic_device() -> int | None:
    for idx, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] <= 0:
            continue
        if any(hint.lower() in d["name"].lower() for hint in VIRTUAL_MIC_NAME_HINTS):
            return idx
    return None


@dataclass
class LatencyStats:
    model_ms: float = 0.0
    total_ms: float = 0.0


class AudioPipeline:
    def __init__(self, torch_device: torch.device):
        self.torch_device = torch_device
        self.effect_chain: EffectChain | None = None
        self.mic_gate: MicNoiseGate | None = None
        self.voice: RVCVoice | None = None
        self.converter: RealtimeConverter | None = None
        self.latency = LatencyStats()
        self.underrun_count = 0
        self.muted = False

        self._in_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._out_resampler: OutputResampleBuffer | None = None
        self._monitor_resampler: OutputResampleBuffer | None = None

        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._monitor_stream: sd.OutputStream | None = None
        self._worker_thread: threading.Thread | None = None
        self._running = threading.Event()

        self._file_feeder_thread: threading.Thread | None = None
        self.on_file_finished = None  # optional callback, set by the server layer
        self.using_file_input = False

    def _open_resampled_output(
        self, device: int, processing_samplerate: int, processing_blocksize: int
    ) -> tuple[sd.OutputStream, OutputResampleBuffer]:
        """Opens an OutputStream at the device's own actual sample rate
        (queried live, not assumed) and returns it paired with a resample
        buffer that converts from our internal processing rate to whatever
        that turns out to be — see OutputResampleBuffer for why this can't
        just be "open at the processing rate and trust the OS to convert."
        """
        try:
            device_info = sd.query_devices(device)
        except Exception as e:
            raise PipelineError("device_not_found", f"Failed to query audio device: {e}") from e
        device_samplerate = int(round(device_info["default_samplerate"]))
        device_blocksize = max(
            1, round(processing_blocksize * device_samplerate / processing_samplerate)
        )
        resampler = OutputResampleBuffer(in_sr=processing_samplerate, out_sr=device_samplerate)
        try:
            stream = sd.OutputStream(
                device=device, channels=1, samplerate=device_samplerate,
                blocksize=device_blocksize, dtype="float32",
                callback=self._make_output_callback(resampler),
            )
        except sd.PortAudioError as e:
            raise PipelineError("device_not_found", f"Failed to open audio device: {e}") from e
        return stream, resampler

    # -- model lifecycle ----------------------------------------------------

    def load_voice(
        self,
        pth_path: str,
        index_path: str | None = None,
        index_rate: float = 0.75,
        f0_up_key: int = 0,
        f0_method: str = "harvest",
        block_time: float = 0.25,
        crossfade_time: float = 0.05,
        extra_time: float = 2.5,
    ) -> None:
        check_cuda_device(self.torch_device)
        try:
            voice = RVCVoice(
                pth_path=pth_path,
                device=self.torch_device,
                index_path=index_path,
                index_rate=index_rate,
            )
        except (ModelLoadError, HubertNotFoundError) as e:
            raise PipelineError("model_not_found", str(e)) from e
        except RuntimeError as e:
            if "CUDA" in str(e) or "no kernel image" in str(e):
                raise PipelineError(
                    "cuda_kernel_mismatch",
                    f"Loading the model crashed with a CUDA error: {e}\n"
                    "Switch to the nightly cu128 torch line — see setup.bat.",
                ) from e
            raise PipelineError("unknown", str(e)) from e

        was_running = self._running.is_set()
        if was_running:
            self.stop()

        self.voice = voice
        self.converter = RealtimeConverter(
            voice,
            self.torch_device,
            block_time=block_time,
            crossfade_time=crossfade_time,
            extra_time=extra_time,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
        )
        self.effect_chain = EffectChain(sample_rate=voice.target_sample_rate)

        if was_running:
            self.start(self._last_input_device, self._last_output_device, self._last_monitor_device)

    def unload_voice(self) -> None:
        was_running = self._running.is_set()
        if was_running:
            self.stop()
        self.voice = None
        self.converter = None
        if was_running:
            self.effect_chain = EffectChain(sample_rate=DEFAULT_SAMPLE_RATE)
            self.start(self._last_input_device, self._last_output_device, self._last_monitor_device)

    def set_voice_pitch(self, f0_up_key: int) -> None:
        if self.converter is not None:
            self.converter.f0_up_key = f0_up_key

    # -- effect control -------------------------------------------------

    def update_effects(self, **kwargs) -> None:
        if "noise_gate_db" in kwargs:
            kwargs = dict(kwargs)
            threshold = kwargs.pop("noise_gate_db")
            if self.mic_gate is None:
                self.mic_gate = MicNoiseGate(sample_rate=DEFAULT_SAMPLE_RATE, threshold_db=threshold)
            else:
                self.mic_gate.set_threshold(threshold)
        if kwargs:
            if self.effect_chain is None:
                self.effect_chain = EffectChain(sample_rate=DEFAULT_SAMPLE_RATE)
            self.effect_chain.update(**kwargs)

    # -- device / stream lifecycle ---------------------------------------

    def start(self, input_device: int, output_device: int, monitor_device: int | None = None) -> None:
        if self._running.is_set():
            # Never open a second set of streams on top of a running one —
            # overlapping PortAudio/CoreAudio streams on the same device can
            # hard-crash the process (not a catchable Python exception).
            self.stop()

        self._last_input_device = input_device
        self._last_output_device = output_device
        self._last_monitor_device = monitor_device

        samplerate = self.converter.target_sample_rate if self.converter else DEFAULT_SAMPLE_RATE
        blocksize = self.converter.block_frame if self.converter else DEFAULT_BLOCKSIZE

        if self.effect_chain is None or self.effect_chain.sample_rate != samplerate:
            self.effect_chain = EffectChain(sample_rate=samplerate)
        if self.mic_gate is None:
            self.mic_gate = MicNoiseGate(sample_rate=samplerate)
        elif self.mic_gate.sample_rate != samplerate:
            self.mic_gate = MicNoiseGate(sample_rate=samplerate, threshold_db=self.mic_gate.threshold_db)

        try:
            self._input_stream = sd.InputStream(
                device=input_device, channels=1, samplerate=samplerate,
                blocksize=blocksize, dtype="float32", callback=self._input_callback,
            )
        except sd.PortAudioError as e:
            raise PipelineError("device_not_found", f"Failed to open audio device: {e}") from e

        self._output_stream, self._out_resampler = self._open_resampled_output(
            output_device, samplerate, blocksize
        )

        if monitor_device is not None:
            self.enable_monitor(monitor_device, samplerate, blocksize)

        # A fresh Event per session, captured by the worker thread as an
        # argument (not read dynamically via self._running) — see
        # OutputResampleBuffer/stop() docs: if start() is called again
        # quickly, the OLD thread must keep observing ITS OWN stop signal
        # even after self._running gets reassigned to a new session's
        # Event, otherwise two worker threads can end up running
        # concurrently, both pulling from the input queue and pushing
        # audio — confirmed empirically as the cause of layered/repeating
        # garbled output.
        running_event = threading.Event()
        running_event.set()
        self._running = running_event

        self._worker_thread = threading.Thread(target=self._worker_loop, args=(running_event,), daemon=True)
        self._worker_thread.start()
        self._input_stream.start()
        self._output_stream.start()

    def start_from_file(
        self, file_path: str, output_device: int, monitor_device: int | None = None
    ) -> None:
        """Streams a WAV file through the exact same conversion pipeline as
        live mic input, instead of capturing from a real microphone. Useful
        for controlled, repeatable testing without background noise or mic
        inconsistency in the way."""
        if self._running.is_set():
            self.stop()

        self.using_file_input = True
        self._last_output_device = output_device
        self._last_monitor_device = monitor_device

        samplerate = self.converter.target_sample_rate if self.converter else DEFAULT_SAMPLE_RATE
        blocksize = self.converter.block_frame if self.converter else DEFAULT_BLOCKSIZE

        if self.effect_chain is None or self.effect_chain.sample_rate != samplerate:
            self.effect_chain = EffectChain(sample_rate=samplerate)
        if self.mic_gate is None:
            self.mic_gate = MicNoiseGate(sample_rate=samplerate)
        elif self.mic_gate.sample_rate != samplerate:
            self.mic_gate = MicNoiseGate(sample_rate=samplerate, threshold_db=self.mic_gate.threshold_db)

        try:
            # librosa.load (not raw soundfile) so compressed formats picked
            # via the native file dialog (mp3/m4a/ogg) decode too, not just
            # WAV/FLAC/AIFF — it falls back to ffmpeg/audioread for those.
            audio, _ = librosa.load(file_path, sr=samplerate, mono=True)
        except Exception as e:
            raise PipelineError(
                "model_not_found",
                f"Failed to read audio file '{file_path}': {e}. If this is an "
                f"mp3/m4a/ogg file, confirm ffmpeg is installed (brew install ffmpeg).",
            ) from e
        audio = audio.astype(np.float32)

        self._output_stream, self._out_resampler = self._open_resampled_output(
            output_device, samplerate, blocksize
        )

        if monitor_device is not None:
            self.enable_monitor(monitor_device, samplerate, blocksize)

        running_event = threading.Event()
        running_event.set()
        self._running = running_event

        self._worker_thread = threading.Thread(target=self._worker_loop, args=(running_event,), daemon=True)
        self._worker_thread.start()
        self._output_stream.start()

        self._file_feeder_thread = threading.Thread(
            target=self._file_feeder_loop, args=(audio, blocksize, samplerate, running_event), daemon=True
        )
        self._file_feeder_thread.start()

    def _file_feeder_loop(
        self, audio: np.ndarray, blocksize: int, samplerate: int, running_event: threading.Event
    ) -> None:
        # The worker thread's output push is non-blocking drop-oldest (by
        # design, for live mic input where hardware naturally paces the
        # input side). A file has no such natural pacing, so without this
        # the worker races through the entire file as fast as the CPU
        # allows and the output queue backpressure alone does NOT slow the
        # feeder down (confirmed empirically — a 90s file finished feeding
        # in ~4s). Pace explicitly against wall-clock time instead.
        pos = 0
        n = len(audio)
        start_time = time.perf_counter()
        while running_event.is_set() and pos < n:
            chunk = audio[pos : pos + blocksize]
            if len(chunk) < blocksize:
                chunk = np.pad(chunk, (0, blocksize - len(chunk)))

            target_elapsed = (pos + blocksize) / samplerate
            actual_elapsed = time.perf_counter() - start_time
            if target_elapsed > actual_elapsed:
                time.sleep(target_elapsed - actual_elapsed)

            while running_event.is_set():
                try:
                    self._in_queue.put(chunk, timeout=0.5)
                    break
                except queue.Full:
                    continue
            pos += blocksize

        if running_event.is_set() and self.on_file_finished is not None:
            self.on_file_finished()

    def enable_monitor(self, monitor_device: int, samplerate: int | None = None, blocksize: int | None = None) -> None:
        samplerate = samplerate or (self.converter.target_sample_rate if self.converter else DEFAULT_SAMPLE_RATE)
        blocksize = blocksize or (self.converter.block_frame if self.converter else DEFAULT_BLOCKSIZE)
        self.disable_monitor()
        self._monitor_stream, self._monitor_resampler = self._open_resampled_output(
            monitor_device, samplerate, blocksize
        )
        self._monitor_stream.start()

    def disable_monitor(self) -> None:
        if self._monitor_stream is not None:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None
            self._monitor_resampler = None

    def stop(self) -> None:
        self._running.clear()
        if self._file_feeder_thread is not None:
            self._file_feeder_thread.join(timeout=2.0)
            self._file_feeder_thread = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                stream.stop()
                stream.close()
        self._input_stream = None
        self._output_stream = None
        self._out_resampler = None
        self.using_file_input = False
        self.disable_monitor()

    # -- callbacks / worker ------------------------------------------------

    def _input_callback(self, indata, frames, time_info, status):
        self._push_nonblocking(self._in_queue, indata[:, 0].copy())

    def _make_output_callback(self, resampler: OutputResampleBuffer):
        def callback(outdata, frames, time_info, status):
            chunk, had_enough = resampler.pop(frames)
            if not had_enough:
                self.underrun_count += 1
            outdata[:, 0] = chunk
        return callback

    def _push_nonblocking(self, q: "queue.Queue[np.ndarray]", chunk: np.ndarray) -> None:
        try:
            q.put_nowait(chunk)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass

    def _worker_loop(self, running_event: threading.Event) -> None:
        while running_event.is_set():
            try:
                mono = self._in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                # Compute the gate's gain from the raw input level, but do
                # NOT apply it to what feeds RVC — the model needs
                # continuous, ungated context across pauses (see
                # MicNoiseGate's docstring for why gating its input broke
                # conversion quality). Apply the gain to the final output
                # instead, further down.
                gate_gain = self.mic_gate.compute_gain(mono) if self.mic_gate is not None else None

                if self.converter is not None:
                    converted = self.converter.process_block(mono)
                    self.latency.model_ms = self.converter.last_infer_ms
                else:
                    converted = mono
                    self.latency.model_ms = 0.0

                if self.effect_chain is not None:
                    converted = self.effect_chain.process(converted)

                if gate_gain is not None:
                    converted = converted * gate_gain
            except Exception:
                import traceback
                traceback.print_exc()
                converted = np.zeros_like(mono)

            self.latency.total_ms = (time.perf_counter() - t0) * 1000

            out_chunk = np.zeros_like(converted) if self.muted else converted
            if self._out_resampler is not None:
                self._out_resampler.push(out_chunk)
            if self._monitor_resampler is not None:
                self._monitor_resampler.push(out_chunk)
