from __future__ import annotations

import logging
import queue
import threading
import time

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
            max_chunk_latency_seconds=self.config.audio_max_chunk_latency_seconds,
            blocksize=self.config.audio_blocksize,
        )
        self._model_presets = self.config.model_presets
        if "sapi" not in {item.strip().lower() for item in self._model_presets}:
            self._model_presets = (*self._model_presets, "sapi")
        self._active_model_index = 0
        initial_primary_model_id, initial_backend = self._resolve_transcriber_preset(
            self._model_presets[self._active_model_index]
        )
        self.transcriber = self._build_transcriber(initial_primary_model_id, initial_backend)

        self._recording = False
        self._text_lock = threading.Lock()
        self._current_text = ""
        self._confirmed_text = ""
        self._last_model_label = ""
        self._session_audio_chunks: list[np.ndarray] = []
        self._dropped_stt_chunks = 0
        self._processed_stt_chunks = 0
        self._stt_stats_counter = 0
        self._buffer_generation = 0
        self._last_mic_log_at = 0.0
        self._mic_last_chunk_at = 0.0
        self._mic_chunks_since_log = 0
        self._mic_samples_since_log = 0
        self._mic_rms_sum = 0.0
        self._mic_peak_max = 0.0
        self._segment_audio: list[np.ndarray] = []
        self._segment_audio_lock = threading.Lock()
        self._silence_samples = 0
        self._silence_threshold_samples = max(
            1, int(self.config.sample_rate * self.config.vad_silence_seconds)
        )
        self._max_segment_samples = max(
            1, int(self.config.sample_rate * self.config.vad_max_segment_seconds)
        )
        self._stt_sequence = 0
        self._stt_input_queue: "queue.Queue[tuple[int, int, np.ndarray]]" = queue.Queue(maxsize=2)
        self._stt_output_queue: "queue.Queue[tuple[int, int, str]]" = queue.Queue()
        self._stop_worker = threading.Event()
        self._worker = threading.Thread(target=self._stt_worker, daemon=True)
        self._worker.start()

        self.hotkeys = HotkeyManager(
            toggle_hotkey=self.config.toggle_hotkey,
            confirm_hotkey=self.config.confirm_hotkey,
            on_toggle=self.toggle_recording,
            on_confirm=self._on_confirm_hotkey,
            switch_model_hotkey=self.config.switch_model_hotkey,
            on_switch_model=self._on_switch_model_hotkey,
            clear_buffer_hotkey=self.config.clear_buffer_hotkey,
            on_clear_buffer=self._on_clear_buffer_hotkey,
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
        LOGGER.info(
            "Audio config: sample_rate=%d chunk=%.2fs max_chunk_latency=%.2fs blocksize=%d",
            self.config.sample_rate,
            self.config.audio_chunk_seconds,
            self.config.audio_max_chunk_latency_seconds,
            self.config.audio_blocksize,
        )
        LOGGER.info(
            (
                "Decode config: realtime beam=%d best_of=%d finalize beam=%d best_of=%d "
                "no_speech=%.2f light_vad=%s min_rms(rt/fn)=%.4f/%.4f max_cps=%.1f"
            ),
            self.config.realtime_beam_size,
            self.config.realtime_best_of,
            self.config.finalize_beam_size,
            self.config.finalize_best_of,
            self.config.whisper_no_speech_threshold,
            self.config.use_light_vad_for_whisper,
            self.config.realtime_min_rms,
            self.config.finalize_min_rms,
            self.config.realtime_max_chars_per_second,
        )
        LOGGER.info(
            "Mic monitor: enabled=%s interval=%.1fs rms_threshold=%.4f",
            self.config.mic_monitor_log_enabled,
            self.config.mic_monitor_log_interval_seconds,
            self.config.mic_monitor_rms_threshold,
        )
        LOGGER.info(
            "VAD segment: silence_rms=%.4f silence_sec=%.1fs max_seg=%.1fs hallucination_rep=%d blacklist=%d items",
            self.config.vad_silence_rms,
            self.config.vad_silence_seconds,
            self.config.vad_max_segment_seconds,
            self.config.hallucination_repetition_threshold,
            len(self.config.hallucination_blacklist),
        )
        LOGGER.info("Active model: %s", self._model_presets[self._active_model_index])
        self.ui.run()

    def _build_transcriber(self, primary_model_id: str, backend: str) -> LocalTranscriber:
        return LocalTranscriber(
            primary_model_id=primary_model_id,
            fallback_model_id=self.config.fallback_model_id,
            cache_dir=self.config.model_cache_dir,
            backend=backend,
            sherpa_onnx_model_dir=self.config.sherpa_onnx_model_dir,
            sherpa_onnx_model_type=self.config.sherpa_onnx_model_type,
            realtime_beam_size=self.config.realtime_beam_size,
            realtime_best_of=self.config.realtime_best_of,
            finalize_beam_size=self.config.finalize_beam_size,
            finalize_best_of=self.config.finalize_best_of,
            whisper_no_speech_threshold=self.config.whisper_no_speech_threshold,
            realtime_condition_on_previous_text=self.config.realtime_condition_on_previous_text,
            finalize_condition_on_previous_text=self.config.finalize_condition_on_previous_text,
            use_light_vad_for_whisper=self.config.use_light_vad_for_whisper,
            realtime_min_rms=self.config.realtime_min_rms,
            finalize_min_rms=self.config.finalize_min_rms,
        )

    def _drop_rate_percent(self) -> float:
        total = self._processed_stt_chunks + self._dropped_stt_chunks
        if total <= 0:
            return 0.0
        return (self._dropped_stt_chunks / float(total)) * 100.0

    @staticmethod
    def _needs_ascii_space(left_char: str, right_char: str) -> bool:
        if not left_char or not right_char:
            return False
        if left_char.isspace() or right_char.isspace():
            return False
        if not left_char.isascii() or not right_char.isascii():
            return False
        return left_char.isalnum() and right_char.isalnum()

    def _merge_realtime_piece(self, base_text: str, new_piece: str) -> str:
        left = base_text.strip()
        right = new_piece.strip()
        if not right:
            return left
        if not left:
            return right

        overlap_limit = max(0, int(self.config.realtime_merge_max_overlap_chars))
        max_overlap = min(overlap_limit, len(left), len(right))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if left[-size:] == right[:size]:
                overlap = size
                break

        merged_piece = right[overlap:]
        if not merged_piece:
            return left

        if self._needs_ascii_space(left[-1], merged_piece[0]):
            return f"{left} {merged_piece}"
        return f"{left}{merged_piece}"

    def _track_mic_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size <= 0:
            return
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        self._mic_chunks_since_log += 1
        self._mic_samples_since_log += int(chunk.size)
        self._mic_rms_sum += rms
        self._mic_peak_max = max(self._mic_peak_max, peak)
        self._mic_last_chunk_at = time.monotonic()

    def _emit_mic_monitor_log(self, force: bool = False) -> None:
        if not self.config.mic_monitor_log_enabled:
            return
        now = time.monotonic()
        interval = max(0.2, float(self.config.mic_monitor_log_interval_seconds))
        if not force and (now - self._last_mic_log_at) < interval:
            return

        if self._mic_chunks_since_log > 0:
            avg_rms = self._mic_rms_sum / float(self._mic_chunks_since_log)
            captured_seconds = self._mic_samples_since_log / float(self.config.sample_rate)
            speech = avg_rms >= float(self.config.mic_monitor_rms_threshold)
            LOGGER.info(
                "MIC monitor: chunks=%d audio=%.2fs avg_rms=%.4f peak=%.4f speech=%s q_in=%d",
                self._mic_chunks_since_log,
                captured_seconds,
                avg_rms,
                self._mic_peak_max,
                "yes" if speech else "no",
                self._stt_input_queue.qsize(),
            )
        else:
            silence_seconds = max(0.0, now - self._mic_last_chunk_at)
            LOGGER.info(
                "MIC monitor: no chunks for %.1fs (recording=%s q_in=%d)",
                silence_seconds,
                self._recording,
                self._stt_input_queue.qsize(),
            )

        self._last_mic_log_at = now
        self._mic_chunks_since_log = 0
        self._mic_samples_since_log = 0
        self._mic_rms_sum = 0.0
        self._mic_peak_max = 0.0

    @staticmethod
    def _realtime_char_count(text: str) -> int:
        return sum(1 for ch in text if not ch.isspace())

    def toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _on_confirm_hotkey(self) -> None:
        self.confirm_and_paste()

    def _on_clear_buffer_hotkey(self) -> None:
        self.clear_recording_buffer()

    def _on_switch_model_hotkey(self) -> None:
        if not self.ui.is_active:
            return
        self.switch_model()

    def _start_recording(self) -> None:
        self._session_audio_chunks = []
        self._dropped_stt_chunks = 0
        self._processed_stt_chunks = 0
        self._stt_stats_counter = 0
        now = time.monotonic()
        self._last_mic_log_at = now
        self._mic_last_chunk_at = now
        self._mic_chunks_since_log = 0
        self._mic_samples_since_log = 0
        self._mic_rms_sum = 0.0
        self._mic_peak_max = 0.0
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0
        with self._text_lock:
            self._confirmed_text = ""
        self.recorder.start()
        self._recording = True
        self.ui.set_status("録音中...")

    def _stop_recording(self) -> None:
        self.recorder.stop()
        self._collect_pending_audio_chunks()
        self._emit_mic_monitor_log(force=True)
        self._recording = False
        self.ui.set_status("待機中")

    def _ensure_mic_off(self) -> None:
        if self._recording:
            self._stop_recording()
            return
        if self.recorder.is_recording:
            self.recorder.stop()
        self._collect_pending_audio_chunks()

    def _schedule_pump(self) -> None:
        self._pump_audio_chunks()
        self._pump_stt_output()
        self._refresh_model_label()
        interval_ms = max(30, int(self.config.ui_pump_interval_ms))
        self.ui.root.after(interval_ms, self._schedule_pump)

    def _current_model_label(self) -> str:
        selected_model = self._model_presets[self._active_model_index]
        if self.transcriber.backend == "faster-whisper":
            return self.transcriber.active_faster_whisper_model_id
        return selected_model

    def _resolve_transcriber_preset(self, preset: str) -> tuple[str, str]:
        normalized = preset.strip()
        if normalized.lower() in {"sherpa-onnx", "sherpa_onnx", "sherpa"}:
            return self.config.primary_model_id, "sherpa-onnx"
        if normalized.lower() in {"sapi", "windows-sapi", "windows_sapi"}:
            return self.config.primary_model_id, "sapi"
        return normalized, self.config.transcriber_backend

    def _refresh_model_label(self, force: bool = False) -> None:
        label = self._current_model_label()
        if force or label != self._last_model_label:
            self.ui.set_model(label)
            self._last_model_label = label

    def _pump_audio_chunks(self) -> None:
        if not self._recording:
            return
        need_enqueue = False
        while True:
            chunk = self.recorder.pop_chunk_nowait()
            if chunk is None:
                break
            if chunk.size == 0:
                continue
            self._track_mic_chunk(chunk)
            self._session_audio_chunks.append(chunk)

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            is_silence = rms < self.config.vad_silence_rms

            with self._segment_audio_lock:
                self._segment_audio.append(chunk)
                seg_samples = sum(c.size for c in self._segment_audio)

            if is_silence:
                self._silence_samples += chunk.size
            else:
                self._silence_samples = 0

            segment_full = seg_samples >= self._max_segment_samples
            silence_exceeded = (
                self._silence_samples >= self._silence_threshold_samples
                and seg_samples > self._silence_threshold_samples
            )

            if segment_full or silence_exceeded:
                self._finalize_current_segment()
            else:
                need_enqueue = True

        if need_enqueue:
            self._enqueue_segment_audio()
        self._emit_mic_monitor_log()

    def _collect_pending_audio_chunks(self) -> None:
        while True:
            chunk = self.recorder.pop_chunk_nowait()
            if chunk is None:
                break
            if chunk.size > 0:
                self._track_mic_chunk(chunk)
                self._session_audio_chunks.append(chunk)
                with self._segment_audio_lock:
                    self._segment_audio.append(chunk)

    def _segment_total_samples(self) -> int:
        with self._segment_audio_lock:
            return sum(c.size for c in self._segment_audio)

    def _get_segment_audio(self) -> np.ndarray:
        with self._segment_audio_lock:
            if not self._segment_audio:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(self._segment_audio)

    def _check_voice_command(self, text: str) -> str | None:
        """Return 'paste', 'clear', or None."""
        stripped = text.strip().rstrip("。．.、，,！!？?")
        LOGGER.info("Voice command check: stripped=%r hex=%s", stripped, stripped.encode("utf-8").hex())
        for cmd in self.config.voice_cmd_paste:
            if cmd and (stripped == cmd or stripped.endswith(cmd)):
                return "paste"
        for cmd in self.config.voice_cmd_clear:
            if cmd and (stripped == cmd or stripped.endswith(cmd)):
                return "clear"
        return None

    def _finalize_current_segment(self) -> None:
        seg_audio = self._get_segment_audio()
        if seg_audio.size == 0:
            with self._segment_audio_lock:
                self._segment_audio = []
            self._silence_samples = 0
            return

        # Run a final transcription on the full segment for accuracy
        try:
            prompt_tail = ""
            with self._text_lock:
                prompt_tail = self._confirmed_text[-self.config.realtime_context_chars :].strip()
            text = self.transcriber.transcribe(
                seg_audio,
                self.config.sample_rate,
                prompt_text=prompt_tail,
                final_pass=False,
            )
            if text and self._is_hallucination(text):
                LOGGER.info("VAD segment finalize filtered (hallucination): %r", text)
                text = ""
            if text:
                voice_cmd = self._check_voice_command(text)
                if voice_cmd == "paste":
                    LOGGER.info("Voice command: paste (text=%r)", text)
                    with self._segment_audio_lock:
                        self._segment_audio = []
                    self._silence_samples = 0
                    with self._text_lock:
                        self._current_text = ""
                    self.ui.root.after(0, self._voice_confirm_and_paste)
                    return
                if voice_cmd == "clear":
                    LOGGER.info("Voice command: clear (text=%r)", text)
                    with self._segment_audio_lock:
                        self._segment_audio = []
                    self._silence_samples = 0
                    self.ui.root.after(0, self.clear_recording_buffer)
                    return
                stripped = text.rstrip()
                if stripped and stripped[-1] not in "。．.！!？?、，,":
                    text = stripped + "。"
                with self._text_lock:
                    self._confirmed_text += text
                LOGGER.info("VAD segment confirmed: %r", text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VAD segment finalize failed: %s", exc)

        # Reset segment
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0
        with self._text_lock:
            self._current_text = ""
            display = self._confirmed_text + self._current_text
        self.ui.set_text(display)

    def _voice_confirm_and_paste(self) -> None:
        """Triggered by voice command — paste without stopping recording."""
        self._buffer_generation += 1
        self._drain_stt_queues()

        with self._text_lock:
            text = (self._confirmed_text + self._current_text).strip()

        LOGGER.info("Voice paste: text=%r", text)

        if not text:
            self.ui.set_status("テキストなし")
            return

        paste_text_to_active_window(text)
        LOGGER.info("Voice paste: done")

        with self._text_lock:
            self._current_text = ""
            self._confirmed_text = ""
        self._session_audio_chunks = []
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0
        self.ui.set_text("")
        self.ui.set_status("音声で貼り付け完了")

    def _enqueue_segment_audio(self) -> None:
        seg_audio = self._get_segment_audio()
        if seg_audio.size == 0:
            return
        generation = self._buffer_generation
        self._stt_sequence += 1
        seq = self._stt_sequence
        # Drop old items to keep only the latest in the queue
        while not self._stt_input_queue.empty():
            try:
                self._stt_input_queue.get_nowait()
                self._dropped_stt_chunks += 1
            except queue.Empty:
                break
        try:
            self._stt_input_queue.put_nowait((generation, seq, seg_audio))
        except queue.Full:
            self._dropped_stt_chunks += 1

    def _stt_worker(self) -> None:
        while not self._stop_worker.is_set():
            try:
                generation, seq, seg_audio = self._stt_input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if seq < self._stt_sequence:
                self._dropped_stt_chunks += 1
                continue
            try:
                prompt_tail = ""
                if self.config.realtime_condition_on_previous_text:
                    with self._text_lock:
                        prompt_tail = self._confirmed_text[-self.config.realtime_context_chars :].strip()
                started_at = time.perf_counter()
                text = self.transcriber.transcribe(
                    seg_audio,
                    self.config.sample_rate,
                    prompt_text=prompt_tail,
                    final_pass=False,
                )
                self._processed_stt_chunks += 1
                elapsed = time.perf_counter() - started_at
                audio_seconds = float(seg_audio.size) / float(self.config.sample_rate)
                rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
                self._stt_stats_counter += 1
                if self._stt_stats_counter % 5 == 0:
                    LOGGER.info(
                        (
                            "STT realtime: model=%s backend=%s seg=%.2fs infer=%.0fms rtf=%.2f "
                            "q_in=%d q_out=%d dropped=%d drop_rate=%.1f%%"
                        ),
                        self._current_model_label(),
                        self.transcriber.backend,
                        audio_seconds,
                        elapsed * 1000.0,
                        rtf,
                        self._stt_input_queue.qsize(),
                        self._stt_output_queue.qsize(),
                        self._dropped_stt_chunks,
                        self._drop_rate_percent(),
                    )
                max_chars_per_second = float(self.config.realtime_max_chars_per_second)
                if text and max_chars_per_second > 0 and audio_seconds > 0:
                    char_rate = self._realtime_char_count(text) / audio_seconds
                    if char_rate > max_chars_per_second:
                        LOGGER.info(
                            "STT realtime filtered (cps): cps=%.1f threshold=%.1f text=%r",
                            char_rate,
                            max_chars_per_second,
                            text,
                        )
                        text = ""
                if text and self._is_hallucination(text):
                    LOGGER.info("STT realtime filtered (hallucination): %r", text)
                    text = ""
                if text:
                    self._stt_output_queue.put((generation, seq, text))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Transcription failed: %s", exc)

    def _is_hallucination(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        for bl in self.config.hallucination_blacklist:
            if bl and stripped == bl:
                return True
        rep_threshold = max(1, self.config.hallucination_repetition_threshold)
        if len(stripped) >= 4:
            for seg_len in range(2, len(stripped) // rep_threshold + 1):
                segment = stripped[:seg_len]
                if segment * rep_threshold == stripped[: seg_len * rep_threshold]:
                    if seg_len * rep_threshold >= len(stripped) * 0.7:
                        return True
        return False

    def _pump_stt_output(self) -> None:
        latest_text = None
        latest_gen = None
        while True:
            try:
                generation, seq, text = self._stt_output_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._buffer_generation:
                continue
            latest_gen = generation
            latest_text = text

        if latest_text is not None and latest_gen == self._buffer_generation:
            with self._text_lock:
                self._current_text = latest_text
                display = self._confirmed_text + self._current_text
            self.ui.set_text(display)

    def confirm_and_paste(self) -> None:
        self._ensure_mic_off()
        self._buffer_generation += 1
        self._drain_stt_queues()

        if self.config.enable_finalize_pass:
            finalized = self._run_finalize_pass()
            if finalized:
                with self._text_lock:
                    self._confirmed_text = ""
                    self._current_text = finalized
                self.ui.set_text(finalized)

        with self._text_lock:
            text = (self._confirmed_text + self._current_text).strip()

        if not text:
            return

        paste_text_to_active_window(text)

        with self._text_lock:
            self._current_text = ""
            self._confirmed_text = ""
        self._session_audio_chunks = []
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0
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
            "STT finalize: model=%s backend=%s audio=%.2fs infer=%.0fms rtf=%.2f dropped=%d drop_rate=%.1f%%",
            self._current_model_label(),
            self.transcriber.backend,
            audio_seconds,
            elapsed * 1000.0,
            rtf,
            self._dropped_stt_chunks,
            self._drop_rate_percent(),
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
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0

        try:
            self.transcriber = self._build_transcriber(next_primary_model_id, next_backend)
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
        self._buffer_generation += 1
        self._drain_stt_queues()
        while self.recorder.pop_chunk_nowait() is not None:
            pass
        self._session_audio_chunks = []
        with self._segment_audio_lock:
            self._segment_audio = []
        self._silence_samples = 0

        with self._text_lock:
            self._current_text = ""
            self._confirmed_text = ""
        self.ui.set_text("")
        if self._recording:
            self.ui.set_status("録音中...（バッファクリア）")
        else:
            self.ui.set_status("待機中（バッファクリア）")

    def _on_close(self) -> None:
        self.hotkeys.unregister()
        if self._recording:
            self.recorder.stop()
        self._stop_worker.set()
        self.ui.root.destroy()
