from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _parse_model_presets(primary_model_id: str) -> tuple[str, ...]:
    raw = os.getenv("MODEL_PRESETS", "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if primary_model_id not in values:
        values.insert(0, primary_model_id)
    return tuple(values)


@dataclass(frozen=True)
class AppConfig:
    primary_model_id: str = os.getenv("PRIMARY_MODEL_ID", "openai/whisper-medium")
    model_presets: tuple[str, ...] = _parse_model_presets(primary_model_id)
    fallback_model_id: str = os.getenv("FALLBACK_MODEL_ID", "Systran/faster-whisper-medium")
    transcriber_backend: str = os.getenv("TRANSCRIBER_BACKEND", "auto")
    sherpa_onnx_model_dir: Path = Path(os.getenv("SHERPA_ONNX_MODEL_DIR", ".models/reazonspeech-v2-sherpa-onnx"))
    sherpa_onnx_model_type: str = os.getenv("SHERPA_ONNX_MODEL_TYPE", "paraformer")
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", ".models"))
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("CHANNELS", "1"))
    audio_chunk_seconds: float = float(os.getenv("AUDIO_CHUNK_SECONDS", "0.6"))
    stt_input_queue_max: int = int(os.getenv("STT_INPUT_QUEUE_MAX", "0"))
    realtime_context_chars: int = int(os.getenv("REALTIME_CONTEXT_CHARS", "80"))
    enable_finalize_pass: bool = os.getenv("ENABLE_FINALIZE_PASS", "1") in {"1", "true", "True"}
    toggle_hotkey: str = os.getenv("TOGGLE_HOTKEY", "ctrl+alt+v")
    confirm_hotkey: str = os.getenv("CONFIRM_HOTKEY", "enter")
    switch_model_hotkey: str = os.getenv("SWITCH_MODEL_HOTKEY", "ctrl+alt+m")
    clear_buffer_hotkey: str = os.getenv("CLEAR_BUFFER_HOTKEY", "esc")
    preview_window_title: str = os.getenv("PREVIEW_WINDOW_TITLE", "KoeFlow preview")
