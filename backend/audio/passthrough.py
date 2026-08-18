"""
Stage 1: mic -> virtual mic pass-through, with an optional monitor output.

Uses one InputStream and up to two OutputStreams (virtual mic + monitor),
connected by small queues, so the input callback never blocks on anything
except a non-blocking queue put. No DSP/ML happens here yet.
"""
import queue
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
CHANNELS = 1
# 10ms (480 samples) proved too tight a deadline for Python-level per-chunk
# handling under normal OS scheduling and caused real audible underruns;
# ~43ms is still low-latency but far more robust.
BLOCKSIZE = 2048
QUEUE_MAXSIZE = 16


class PassthroughEngine:
    def __init__(
        self,
        input_device: int,
        output_device: int,
        monitor_device: int | None = None,
        samplerate: int = SAMPLE_RATE,
        blocksize: int = BLOCKSIZE,
        channels: int = CHANNELS,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels

        self._out_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._monitor_queue: queue.Queue[np.ndarray] | None = None
        self._monitor_lock = threading.Lock()

        self._underrun_count = 0

        self._input_stream = sd.InputStream(
            device=input_device,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            device=output_device,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=self._make_output_callback(self._out_queue),
        )
        self._monitor_stream: sd.OutputStream | None = None
        if monitor_device is not None:
            self.enable_monitor(monitor_device)

    def _input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[input] status: {status}")
        chunk = indata.copy()
        self._push_nonblocking(self._out_queue, chunk)
        with self._monitor_lock:
            if self._monitor_queue is not None:
                self._push_nonblocking(self._monitor_queue, chunk)

    def _push_nonblocking(self, q: "queue.Queue[np.ndarray]", chunk: np.ndarray) -> None:
        try:
            q.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest block rather than blocking the audio callback.
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass

    def _make_output_callback(self, q: "queue.Queue[np.ndarray]"):
        def callback(outdata, frames, time_info, status):
            if status:
                print(f"[output] status: {status}")
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                self._underrun_count += 1
                return
            outdata[:] = chunk

        return callback

    def enable_monitor(self, monitor_device: int) -> None:
        with self._monitor_lock:
            if self._monitor_stream is not None:
                self._monitor_stream.stop()
                self._monitor_stream.close()
            self._monitor_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            self._monitor_stream = sd.OutputStream(
                device=monitor_device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._make_output_callback(self._monitor_queue),
            )
            self._monitor_stream.start()

    def disable_monitor(self) -> None:
        with self._monitor_lock:
            if self._monitor_stream is not None:
                self._monitor_stream.stop()
                self._monitor_stream.close()
                self._monitor_stream = None
                self._monitor_queue = None

    def start(self) -> None:
        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> None:
        self._input_stream.stop()
        self._input_stream.close()
        self._output_stream.stop()
        self._output_stream.close()
        self.disable_monitor()

    @property
    def underrun_count(self) -> int:
        return self._underrun_count
