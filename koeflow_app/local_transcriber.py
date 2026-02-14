from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download


LOGGER = logging.getLogger(__name__)


class LocalTranscriber:
    def __init__(
        self,
        primary_model_id: str,
        fallback_model_id: str,
        cache_dir: Path,
        backend: str = "auto",
        sherpa_onnx_model_dir: Path | None = None,
        sherpa_onnx_model_type: str = "paraformer",
    ) -> None:
        self.primary_model_id = primary_model_id
        self.fallback_model_id = fallback_model_id
        self.cache_dir = cache_dir
        self.backend_preference = backend.strip().lower()
        self.sherpa_onnx_model_dir = sherpa_onnx_model_dir or (self.cache_dir / "reazonspeech-v2-sherpa-onnx")
        self.sherpa_onnx_model_type = sherpa_onnx_model_type.strip().lower()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        configured_primary_path = Path(self.primary_model_id)
        self._primary_is_existing_local_path = configured_primary_path.exists()
        self._primary_local_dir = (
            configured_primary_path
            if self._primary_is_existing_local_path
            else self.cache_dir / "primary" / self.primary_model_id.replace("/", "--")
        )
        self._fallback_download_root = self.cache_dir / "fallback"

        self.backend = "faster-whisper"
        self._primary_pipe = None
        self._fallback_whisper: Optional[WhisperModel] = None
        self._sherpa_onnx_recognizer: Any = None

        if self.backend_preference in {"sherpa-onnx", "sherpa_onnx", "sherpa"}:
            if self._try_load_sherpa_onnx():
                return
            LOGGER.warning("Sherpa-ONNX load failed. Falling back to auto backend selection.")

        if self.backend_preference == "faster-whisper":
            self._load_fallback_whisper()
            return

        self._try_load_primary_transformers()
        if self._primary_pipe is None:
            self._load_fallback_whisper()

    def _find_first_existing(self, base_dir: Path, candidates: tuple[str, ...]) -> Optional[Path]:
        for name in candidates:
            path = base_dir / name
            if path.exists():
                return path
        return None

    def _try_load_sherpa_onnx(self) -> bool:
        try:
            import sherpa_onnx
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("sherpa-onnx import failed: %s", exc)
            return False

        model_dir = self.sherpa_onnx_model_dir
        tokens = self._find_first_existing(model_dir, ("tokens.txt", "tokens"))
        if tokens is None:
            LOGGER.warning("Sherpa-ONNX tokens file not found in %s", model_dir)
            return False

        if self.sherpa_onnx_model_type == "paraformer":
            paraformer = self._find_first_existing(model_dir, ("model.int8.onnx", "model.onnx", "paraformer.onnx"))
            if paraformer is None:
                LOGGER.warning("Sherpa-ONNX paraformer model not found in %s", model_dir)
                return False

            try:
                self._sherpa_onnx_recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=str(paraformer),
                    tokens=str(tokens),
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    provider="cpu",
                )
                self.backend = "sherpa-onnx"
                LOGGER.info("Primary model loaded with sherpa-onnx paraformer: %s", paraformer)
                return True
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Sherpa-ONNX recognizer init failed (%s)", exc)
                self._sherpa_onnx_recognizer = None
                return False

        if self.sherpa_onnx_model_type in {"zipformer-transducer", "zipformer_transducer", "zipformer", "transducer"}:
            encoder = self._find_first_existing(
                model_dir,
                ("encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx", "encoder.int8.onnx", "encoder.onnx"),
            )
            decoder = self._find_first_existing(
                model_dir,
                ("decoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx", "decoder.int8.onnx", "decoder.onnx"),
            )
            joiner = self._find_first_existing(
                model_dir,
                ("joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx", "joiner.int8.onnx", "joiner.onnx"),
            )

            if encoder is None or decoder is None or joiner is None:
                LOGGER.warning("Sherpa-ONNX transducer files not found in %s", model_dir)
                return False

            try:
                self._sherpa_onnx_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(encoder),
                    decoder=str(decoder),
                    joiner=str(joiner),
                    tokens=str(tokens),
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    provider="cpu",
                    model_type="transducer",
                )
                self.backend = "sherpa-onnx"
                LOGGER.info("Primary model loaded with sherpa-onnx transducer: %s", encoder)
                return True
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Sherpa-ONNX transducer init failed (%s)", exc)
                self._sherpa_onnx_recognizer = None
                return False

        LOGGER.warning(
            "Unsupported SHERPA_ONNX_MODEL_TYPE=%s (supported: paraformer, zipformer-transducer)",
            self.sherpa_onnx_model_type,
        )
        return False

    def _try_load_primary_transformers(self) -> None:
        try:
            from transformers import pipeline

            needs_download = not self._is_complete_transformers_model_dir(self._primary_local_dir)
            if needs_download and not self._primary_is_existing_local_path:
                if self._primary_local_dir.exists():
                    LOGGER.warning("Primary cache looks incomplete. Re-downloading: %s", self._primary_local_dir)
                    shutil.rmtree(self._primary_local_dir, ignore_errors=True)
                snapshot_download(
                    repo_id=self.primary_model_id,
                    local_dir=str(self._primary_local_dir),
                )

            self._primary_pipe = pipeline(
                task="automatic-speech-recognition",
                model=str(self._primary_local_dir),
                device=-1,
                trust_remote_code=True,
            )
            self.backend = "transformers"
            LOGGER.info("Primary model loaded with transformers: %s", self.primary_model_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Primary model load failed (%s). Fallback to faster-whisper.", exc)
            self._primary_pipe = None

    def _is_complete_transformers_model_dir(self, model_dir: Path) -> bool:
        if not model_dir.exists():
            return False
        if not (model_dir / "config.json").exists():
            return False
        if any(
            (model_dir / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack")
        ):
            return True
        if (model_dir / "model.safetensors.index.json").exists() or (model_dir / "pytorch_model.bin.index.json").exists():
            return True
        if list(model_dir.glob("model-*.safetensors")):
            return True
        return False

    def _load_fallback_whisper(self) -> None:
        self._fallback_whisper = WhisperModel(
            self.fallback_model_id,
            device="cpu",
            compute_type="int8",
            download_root=str(self._fallback_download_root),
        )
        self.backend = "faster-whisper"
        LOGGER.info("Fallback model loaded with faster-whisper: %s", self.fallback_model_id)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        audio_f32 = audio.astype(np.float32)
        # Skip near-silent chunks to avoid common Whisper hallucinations
        rms = float(np.sqrt(np.mean(np.square(audio_f32))))
        if rms < 0.003:
            return ""

        if self.backend == "sherpa-onnx" and self._sherpa_onnx_recognizer is not None:
            try:
                stream = self._sherpa_onnx_recognizer.create_stream()
                stream.accept_waveform(sample_rate, audio_f32)
                self._sherpa_onnx_recognizer.decode_stream(stream)
                text = ""
                if hasattr(stream, "result"):
                    result = stream.result
                    text = getattr(result, "text", str(result))
                elif hasattr(self._sherpa_onnx_recognizer, "get_result"):
                    result = self._sherpa_onnx_recognizer.get_result(stream)
                    text = getattr(result, "text", str(result))
                return text.strip()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Sherpa-ONNX transcribe failed (%s). Switching fallback.", exc)
                self._sherpa_onnx_recognizer = None
                self._load_fallback_whisper()

        if self.backend == "transformers" and self._primary_pipe is not None:
            try:
                result = self._primary_pipe(
                    {"array": audio_f32, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "ja",
                        "temperature": 0.0,
                        "condition_on_prev_tokens": False,
                    },
                )
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                return text.strip()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Primary transcribe failed (%s). Switching fallback.", exc)
                self._primary_pipe = None
                self._load_fallback_whisper()

        if self._fallback_whisper is None:
            self._load_fallback_whisper()

        segments, _ = self._fallback_whisper.transcribe(
            audio_f32,
            language="ja",
            vad_filter=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()
