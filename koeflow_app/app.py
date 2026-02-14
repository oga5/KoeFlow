from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from .audio_capture import AudioCapture
from .config import AppConfig
from .hotkeys import HotkeyManager
from .local_transcriber import LocalTranscriber
from .overlay_ui import OverlayUI
from .paste_target import paste_text_to_active_window


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class VoiceInputApp:
    def __init__(self) -> None:
        self.config = AppConfig()
        self.ui = OverlayUI(
            title=self.config.preview_window_title,
            toggle_hotkey=self.config.toggle_hotkey,
            confirm_hotkey=self.config.confirm_hotkey,
            switch_model_hotkey=self.config.switch_model_hotkey,
            clear_buffer_hotkey=self.config.clear_buffer_hotkey,
        )
        self.recorder = AudioCapture(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_seconds=self.config.audio_chunk_seconds,
        )
        self._model_presets = self.config.model_presets
        self._active_model_index = 0
        initial_primary_model_id, initial_backend = self._resolve_transcriber_preset(
            self._model_presets[self._active_model_index]
        )
        self.transcriber = LocalTranscriber(
            primary_model_id=initial_primary_model_id,
            fallback_model_id=self.config.fallback_model_id,
            cache_dir=self.config.model_cache_dir,
            backend=initial_backend,
            sherpa_onnx_model_dir=self.config.sherpa_onnx_model_dir,
            sherpa_onnx_model_type=self.config.sherpa_onnx_model_type,
        )

        self._recording = False
        self._text_lock = threading.Lock()
        self._current_text = ""
        self._last_model_label = ""
        self._session_audio_chunks: list[np.ndarray] = []
        self._dropped_stt_chunks = 0
        self._stt_stats_counter = 0
        self._buffer_generation = 0
        queue_maxsize = int(self.config.stt_input_queue_max)
        self._stt_input_queue: "queue.Queue[tuple[int, np.ndarray]]" = (
            queue.Queue() if queue_maxsize <= 0 else queue.Queue(maxsize=queue_maxsize)
        )
        self._stt_output_queue: "queue.Queue[tuple[int, str]]" = queue.Queue()
        self._stop_worker = threading.Event()
        self._worker = threading.Thread(target=self._stt_worker, daemon=True)
        self._worker.start()

        self.hotkeys = HotkeyManager(
            toggle_hotkey=self.config.toggle_hotkey,
            confirm_hotkey=self.config.confirm_hotkey,
            on_toggle=self.toggle_recording,
            on_confirm=self.confirm_and_paste,
            switch_model_hotkey=self.config.switch_model_hotkey,
            on_switch_model=self.switch_model,
            clear_buffer_hotkey=self.config.clear_buffer_hotkey,
            on_clear_buffer=self.clear_recording_buffer,
        )

        self.ui.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.hotkeys.register()
        self.ui.set_status("待機中")
        self._refresh_model_label(force=True)
        self._schedule_pump()
        LOGGER.info(
            "Hotkeys: toggle=%s, confirm=%s, switch-model=%s, clear-buffer=%s",
            self.config.toggle_hotkey,
            self.config.confirm_hotkey,
            self.config.switch_model_hotkey,
            self.config.clear_buffer_hotkey,
        )
        LOGGER.info("Active model: %s", self._model_presets[self._active_model_index])
        self.ui.run()

    def toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        self._session_audio_chunks = []
        self.recorder.start()
        self._recording = True
        self.ui.set_status("録音中...")

    def _stop_recording(self) -> None:
        self.recorder.stop()
        self._collect_pending_audio_chunks()
        self._recording = False
        self.ui.set_status("待機中")

    def _schedule_pump(self) -> None:
        self._pump_audio_chunks()
        self._pump_stt_output()
        self._refresh_model_label()
        self.ui.root.after(200, self._schedule_pump)

    def _current_model_label(self) -> str:
        selected_model = self._model_presets[self._active_model_index]
        if self.transcriber.backend == "faster-whisper":
            return self.config.fallback_model_id
        return selected_model

    def _resolve_transcriber_preset(self, preset: str) -> tuple[str, str]:
        normalized = preset.strip()
        if normalized.lower() in {"sherpa-onnx", "sherpa_onnx", "sherpa"}:
            return self.config.primary_model_id, "sherpa-onnx"
        return normalized, self.config.transcriber_backend

    def _refresh_model_label(self, force: bool = False) -> None:
        label = self._current_model_label()
        if force or label != self._last_model_label:
            self.ui.set_model(label)
            self._last_model_label = label

    def _pump_audio_chunks(self) -> None:
        if not self._recording:
            return
        generation = self._buffer_generation
        while True:
            chunk = self.recorder.pop_chunk_nowait()
            if chunk is None:
                break
            if chunk.size > 0:
                self._session_audio_chunks.append(chunk)
                self._enqueue_stt_chunk(generation, chunk)

    def _collect_pending_audio_chunks(self) -> None:
        while True:
            chunk = self.recorder.pop_chunk_nowait()
            if chunk is None:
                break
            if chunk.size > 0:
                self._session_audio_chunks.append(chunk)

    def _enqueue_stt_chunk(self, generation: int, chunk: np.ndarray) -> None:
        if self._stt_input_queue.full():
            self._dropped_stt_chunks += 1
            return
        try:
            self._stt_input_queue.put_nowait((generation, chunk))
        except queue.Full:
            self._dropped_stt_chunks += 1

    def _stt_worker(self) -> None:
        while not self._stop_worker.is_set():
            try:
                generation, chunk = self._stt_input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self._text_lock:
                    prompt_tail = self._current_text[-self.config.realtime_context_chars :].strip()
                started_at = time.perf_counter()
                text = self.transcriber.transcribe(
                    chunk,
                    self.config.sample_rate,
                    prompt_text=prompt_tail,
                    final_pass=False,
                )
                elapsed = time.perf_counter() - started_at
                chunk_seconds = float(chunk.size) / float(self.config.sample_rate)
                rtf = elapsed / chunk_seconds if chunk_seconds > 0 else 0.0
                self._stt_stats_counter += 1
                if self._stt_stats_counter % 10 == 0:
                    LOGGER.info(
                        "STT stats: chunk=%.2fs infer=%.0fms rtf=%.2f q_in=%d q_out=%d dropped=%d",
                        chunk_seconds,
                        elapsed * 1000.0,
                        rtf,
                        self._stt_input_queue.qsize(),
                        self._stt_output_queue.qsize(),
                        self._dropped_stt_chunks,
                    )
                if text:
                    self._stt_output_queue.put((generation, text))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Transcription failed: %s", exc)

    def _pump_stt_output(self) -> None:
        updated = False
        while True:
            try:
                generation, piece = self._stt_output_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._buffer_generation:
                continue
            with self._text_lock:
                self._current_text = f"{self._current_text} {piece}".strip()
            updated = True

        if updated:
            with self._text_lock:
                text = self._current_text
            self.ui.set_text(text)

    def confirm_and_paste(self) -> None:
        if self._recording:
            self._stop_recording()

        if self.config.enable_finalize_pass:
            finalized = self._run_finalize_pass()
            if finalized:
                with self._text_lock:
                    self._current_text = finalized
                self.ui.set_text(finalized)

        with self._text_lock:
            text = self._current_text.strip()

        if not text:
            return

        paste_text_to_active_window(text)

        with self._text_lock:
            self._current_text = ""
        self._session_audio_chunks = []
        self.ui.set_text("")
        self.ui.set_status("貼り付け完了")

    def _run_finalize_pass(self) -> str:
        if not self._session_audio_chunks:
            return ""
        merged_audio = np.concatenate(self._session_audio_chunks)
        started_at = time.perf_counter()
        try:
            finalized = self.transcriber.transcribe(
                merged_audio,
                self.config.sample_rate,
                prompt_text="",
                final_pass=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Finalize pass failed: %s", exc)
            return ""
        elapsed = time.perf_counter() - started_at
        audio_seconds = float(merged_audio.size) / float(self.config.sample_rate)
        rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
        LOGGER.info(
            "Finalize pass: audio=%.2fs infer=%.0fms rtf=%.2f",
            audio_seconds,
            elapsed * 1000.0,
            rtf,
        )
        return finalized.strip()

    def switch_model(self) -> None:
        if len(self._model_presets) <= 1:
            self.ui.set_status("モデル候補が1つのみです")
            return

        if self._recording:
            self._stop_recording()

        self._active_model_index = (self._active_model_index + 1) % len(self._model_presets)
        next_model = self._model_presets[self._active_model_index]
        next_primary_model_id, next_backend = self._resolve_transcriber_preset(next_model)
        self.ui.set_status(f"モデル切替中: {next_model}")
        self._buffer_generation += 1
        self._drain_stt_queues()
        self._session_audio_chunks = []

        try:
            self.transcriber = LocalTranscriber(
                primary_model_id=next_primary_model_id,
                fallback_model_id=self.config.fallback_model_id,
                cache_dir=self.config.model_cache_dir,
                backend=next_backend,
                sherpa_onnx_model_dir=self.config.sherpa_onnx_model_dir,
                sherpa_onnx_model_type=self.config.sherpa_onnx_model_type,
            )
            LOGGER.info("Switched model: %s", next_model)
            self._refresh_model_label(force=True)
            self.ui.set_status(f"モデル切替完了: {next_model}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Model switch failed (%s)", exc)
            self.ui.set_status(f"モデル切替失敗: {next_model}")

    def _drain_stt_queues(self) -> None:
        while True:
            try:
                self._stt_input_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._stt_output_queue.get_nowait()
            except queue.Empty:
                break

    def clear_recording_buffer(self) -> None:
        if not self._recording:
            self.ui.set_status("録音中のみバッファクリアできます")
            return

        self._buffer_generation += 1
        self._drain_stt_queues()
        while self.recorder.pop_chunk_nowait() is not None:
            pass
        self._session_audio_chunks = []

        with self._text_lock:
            self._current_text = ""
        self.ui.set_text("")
        self.ui.set_status("録音中...（バッファクリア）")

    def _on_close(self) -> None:
        self.hotkeys.unregister()
        if self._recording:
            self.recorder.stop()
        self._stop_worker.set()
        self.ui.root.destroy()
