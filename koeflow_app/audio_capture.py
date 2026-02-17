from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        chunk_seconds: float = 0.3,
        max_chunk_latency_seconds: float = 0.0,
        blocksize: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = max(0.05, chunk_seconds)
        max_latency = float(max_chunk_latency_seconds)
        if max_latency <= 0.0:
            max_latency = self.chunk_seconds
        self.max_chunk_latency_seconds = max(0.05, max_latency)
        self.blocksize = max(1, int(blocksize))
        self._target_chunk_samples = max(1, int(self.sample_rate * self.chunk_seconds))
        self._max_chunk_latency_samples = max(
            1,
            int(self.sample_rate * min(self.chunk_seconds, self.max_chunk_latency_seconds)),
        )
        self._pending_audio = np.empty(0, dtype=np.float32)
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._recording = False
        self._chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True

        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break

        self._pending_audio = np.empty(0, dtype=np.float32)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
            blocksize=self.blocksize,
        )
        self._stream.start()

    def stop(self) -> None:
        with self._lock:
            self._recording = False
            if self._pending_audio.size > 0:
                self._chunk_queue.put(self._pending_audio.copy())
                self._pending_audio = np.empty(0, dtype=np.float32)

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time, status) -> None:  # noqa: ANN001
        if status:
            return
        with self._lock:
            if not self._recording:
                return
            chunk = np.squeeze(indata.copy())
            if chunk.ndim > 1:
                chunk = chunk[:, 0]
            chunk_f32 = chunk.astype(np.float32) / 32768.0
            self._pending_audio = np.concatenate((self._pending_audio, chunk_f32))
            while self._pending_audio.size >= self._target_chunk_samples:
                ready = self._pending_audio[: self._target_chunk_samples]
                self._pending_audio = self._pending_audio[self._target_chunk_samples :]
                self._chunk_queue.put(ready)
            if self._pending_audio.size >= self._max_chunk_latency_samples:
                self._chunk_queue.put(self._pending_audio.copy())
                self._pending_audio = np.empty(0, dtype=np.float32)

    def pop_chunk_nowait(self) -> Optional[np.ndarray]:
        try:
            return self._chunk_queue.get_nowait()
        except queue.Empty:
            return None
