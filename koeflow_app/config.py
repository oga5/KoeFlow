from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _parse_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in {"1", "true", "True"}


def _parse_model_presets(primary_model_id: str) -> tuple[str, ...]:
    raw = os.getenv("MODEL_PRESETS", "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if primary_model_id not in values:
        values.insert(0, primary_model_id)
    return tuple(values)


@dataclass(frozen=True)
class AppConfig:
    primary_model_id: str = os.getenv("PRIMARY_MODEL_ID", "Systran/faster-whisper-medium")
    model_presets: tuple[str, ...] = _parse_model_presets(primary_model_id)
    fallback_model_id: str = os.getenv("FALLBACK_MODEL_ID", "Systran/faster-whisper-medium")
    transcriber_backend: str = os.getenv("TRANSCRIBER_BACKEND", "faster-whisper")
    sherpa_onnx_model_dir: Path = Path(os.getenv("SHERPA_ONNX_MODEL_DIR", ".models/reazonspeech-v2-sherpa-onnx"))
    sherpa_onnx_model_type: str = os.getenv("SHERPA_ONNX_MODEL_TYPE", "paraformer")
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", ".models"))
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "48000"))
    channels: int = int(os.getenv("CHANNELS", "1"))
    audio_blocksize: int = int(os.getenv("AUDIO_BLOCKSIZE", "1024"))
    audio_chunk_seconds: float = float(os.getenv("AUDIO_CHUNK_SECONDS", "0.6"))
    stt_input_queue_max: int = int(os.getenv("STT_INPUT_QUEUE_MAX", "0"))
    realtime_context_chars: int = int(os.getenv("REALTIME_CONTEXT_CHARS", "80"))
    realtime_max_chars_per_second: float = float(os.getenv("REALTIME_MAX_CHARS_PER_SECOND", "14"))
    realtime_merge_max_overlap_chars: int = int(os.getenv("REALTIME_MERGE_MAX_OVERLAP_CHARS", "12"))
    realtime_beam_size: int = int(os.getenv("REALTIME_BEAM_SIZE", "1"))
    realtime_best_of: int = int(os.getenv("REALTIME_BEST_OF", "1"))
    finalize_beam_size: int = int(os.getenv("FINALIZE_BEAM_SIZE", "5"))
    finalize_best_of: int = int(os.getenv("FINALIZE_BEST_OF", "5"))
    whisper_no_speech_threshold: float = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.75"))
    use_light_vad_for_whisper: bool = _parse_bool_env("USE_LIGHT_VAD_FOR_WHISPER", "1")
    realtime_min_rms: float = float(os.getenv("REALTIME_MIN_RMS", "0.010"))
    finalize_min_rms: float = float(os.getenv("FINALIZE_MIN_RMS", "0.003"))
    realtime_condition_on_previous_text: bool = _parse_bool_env("REALTIME_CONDITION_ON_PREVIOUS_TEXT", "0")
    finalize_condition_on_previous_text: bool = _parse_bool_env("FINALIZE_CONDITION_ON_PREVIOUS_TEXT", "1")
    enable_finalize_pass: bool = _parse_bool_env("ENABLE_FINALIZE_PASS", "1")
    mic_monitor_log_enabled: bool = _parse_bool_env("MIC_MONITOR_LOG_ENABLED", "1")
    mic_monitor_log_interval_seconds: float = float(os.getenv("MIC_MONITOR_LOG_INTERVAL_SECONDS", "2.0"))
    mic_monitor_rms_threshold: float = float(os.getenv("MIC_MONITOR_RMS_THRESHOLD", "0.008"))
    toggle_hotkey: str = os.getenv("TOGGLE_HOTKEY", "ctrl+alt+v")
    confirm_hotkey: str = os.getenv("CONFIRM_HOTKEY", "enter")
    switch_model_hotkey: str = os.getenv("SWITCH_MODEL_HOTKEY", "ctrl+alt+m")
    clear_buffer_hotkey: str = os.getenv("CLEAR_BUFFER_HOTKEY", "esc")
    preview_window_title: str = os.getenv("PREVIEW_WINDOW_TITLE", "KoeFlow preview")
