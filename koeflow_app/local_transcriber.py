from __future__ import annotations

import logging
import subprocess
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download


LOGGER = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 16000
LIGHT_VAD_FRAME_SECONDS = 0.03
LIGHT_VAD_PAD_SECONDS = 0.12
LIGHT_VAD_ENERGY_THRESHOLD = 0.004
LIGHT_VAD_MIN_SPEECH_SECONDS = 0.15
PRIMARY_TRANSFORMERS_REDOWNLOAD_LIMIT = 2
VAD_FILTER_MIN_AUDIO_SECONDS = 2.0


class LocalTranscriber:
    def __init__(
        self,
        primary_model_id: str,
        fallback_model_id: str,
        cache_dir: Path,
        backend: str = "auto",
        sherpa_onnx_model_dir: Path | None = None,
        sherpa_onnx_model_type: str = "paraformer",
        realtime_beam_size: int = 2,
        realtime_best_of: int = 2,
        finalize_beam_size: int = 5,
        finalize_best_of: int = 5,
        whisper_no_speech_threshold: float = 0.6,
        realtime_condition_on_previous_text: bool = False,
        finalize_condition_on_previous_text: bool = True,
        use_light_vad_for_whisper: bool = True,
        realtime_min_rms: float = 0.008,
        finalize_min_rms: float = 0.003,
    ) -> None:
        self.primary_model_id = primary_model_id
        self.fallback_model_id = fallback_model_id
        self.cache_dir = cache_dir
        self.backend_preference = backend.strip().lower()
        self.sherpa_onnx_model_dir = sherpa_onnx_model_dir or (self.cache_dir / "reazonspeech-v2-sherpa-onnx")
        self.sherpa_onnx_model_type = sherpa_onnx_model_type.strip().lower()
        self.realtime_beam_size = max(1, int(realtime_beam_size))
        self.realtime_best_of = max(self.realtime_beam_size, int(realtime_best_of))
        self.finalize_beam_size = max(1, int(finalize_beam_size))
        self.finalize_best_of = max(self.finalize_beam_size, int(finalize_best_of))
        self.whisper_no_speech_threshold = float(np.clip(whisper_no_speech_threshold, 0.0, 1.0))
        self.realtime_condition_on_previous_text = bool(realtime_condition_on_previous_text)
        self.finalize_condition_on_previous_text = bool(finalize_condition_on_previous_text)
        self.use_light_vad_for_whisper = bool(use_light_vad_for_whisper)
        self.realtime_min_rms = max(0.0, float(realtime_min_rms))
        self.finalize_min_rms = max(0.0, float(finalize_min_rms))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        configured_primary_path = Path(self.primary_model_id)
        self._primary_is_existing_local_path = configured_primary_path.exists()
        self._primary_local_dir = (
            configured_primary_path
            if self._primary_is_existing_local_path
            else self.cache_dir / "primary" / self.primary_model_id.replace("/", "--")
        )
        self._primary_transformers_redownload_count_file = (
            self.cache_dir / "primary" / f"{self._primary_local_dir.name}.transformers_redownload_count"
        )
        self._fallback_download_root = self.cache_dir / "fallback"

        self.backend = "faster-whisper"
        self._primary_pipe = None
        self._fallback_whisper: Optional[WhisperModel] = None
        self._sherpa_onnx_recognizer: Any = None
        self._active_faster_whisper_model_id = self.fallback_model_id

        if self.backend_preference in {"sherpa-onnx", "sherpa_onnx", "sherpa"}:
            if self._try_load_sherpa_onnx():
                return
            LOGGER.warning("Sherpa-ONNX load failed. Falling back to auto backend selection.")

        if self.backend_preference in {"sapi", "windows-sapi", "windows_sapi"}:
            self.backend = "sapi"
            LOGGER.info("Primary backend set to Windows SAPI")
            return

        if self.backend_preference == "faster-whisper":
            if self.primary_model_id != self.fallback_model_id and self._try_load_primary_faster_whisper():
                return
            self._load_fallback_whisper()
            return

        if self.backend_preference in {"auto", ""} and self._looks_like_faster_whisper_target():
            LOGGER.info("Primary model looks like faster-whisper format; preferring faster-whisper backend.")
            if self._try_load_primary_faster_whisper():
                return

        self._try_load_primary_transformers()
        if self._primary_pipe is None:
            self._load_fallback_whisper()

    def _looks_like_faster_whisper_target(self) -> bool:
        model_id_lower = self.primary_model_id.strip().lower()
        if "faster-whisper" in model_id_lower:
            return True
        if self._is_complete_faster_whisper_model_dir(self._primary_local_dir):
            return True
        return False

    def _is_complete_faster_whisper_model_dir(self, model_dir: Path) -> bool:
        if not model_dir.exists():
            return False
        if not (model_dir / "model.bin").exists():
            return False
        if not (model_dir / "config.json").exists():
            return False
        return True

    def _read_primary_transformers_redownload_count(self) -> int:
        try:
            raw = self._primary_transformers_redownload_count_file.read_text(encoding="utf-8").strip()
            return max(0, int(raw))
        except Exception:  # noqa: BLE001
            return 0

    def _write_primary_transformers_redownload_count(self, count: int) -> None:
        try:
            self._primary_transformers_redownload_count_file.parent.mkdir(parents=True, exist_ok=True)
            self._primary_transformers_redownload_count_file.write_text(str(max(0, int(count))), encoding="utf-8")
        except Exception:  # noqa: BLE001
            return

    def _clear_primary_transformers_redownload_count(self) -> None:
        try:
            if self._primary_transformers_redownload_count_file.exists():
                self._primary_transformers_redownload_count_file.unlink()
        except Exception:  # noqa: BLE001
            return

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
                redownload_count = self._read_primary_transformers_redownload_count()
                if redownload_count >= PRIMARY_TRANSFORMERS_REDOWNLOAD_LIMIT:
                    LOGGER.warning(
                        "Primary cache looks incomplete after %d re-download attempts. "
                        "Skipping re-download: %s",
                        redownload_count,
                        self._primary_local_dir,
                    )
                else:
                    if self._primary_local_dir.exists():
                        LOGGER.warning(
                            "Primary cache looks incomplete. Re-downloading (%d/%d): %s",
                            redownload_count + 1,
                            PRIMARY_TRANSFORMERS_REDOWNLOAD_LIMIT,
                            self._primary_local_dir,
                        )
                        shutil.rmtree(self._primary_local_dir, ignore_errors=True)
                    snapshot_download(
                        repo_id=self.primary_model_id,
                        local_dir=str(self._primary_local_dir),
                    )
                    self._write_primary_transformers_redownload_count(redownload_count + 1)

            self._primary_pipe = pipeline(
                task="automatic-speech-recognition",
                model=str(self._primary_local_dir),
                device=-1,
                trust_remote_code=True,
            )
            self._clear_primary_transformers_redownload_count()
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

    def _try_load_primary_faster_whisper(self) -> bool:
        try:
            self._load_whisper_model(self.primary_model_id, model_role="Primary")
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Primary faster-whisper model load failed (%s). Falling back to %s.",
                exc,
                self.fallback_model_id,
            )
            return False

    def _load_whisper_model(self, model_id: str, model_role: str) -> None:
        self._fallback_whisper = WhisperModel(
            model_id,
            device="cpu",
            compute_type="int8",
            download_root=str(self._fallback_download_root),
        )
        self._active_faster_whisper_model_id = model_id
        self.backend = "faster-whisper"
        LOGGER.info(
            "%s model loaded with faster-whisper: %s (realtime beam=%d best_of=%d, finalize beam=%d best_of=%d)",
            model_role,
            model_id,
            self.realtime_beam_size,
            self.realtime_best_of,
            self.finalize_beam_size,
            self.finalize_best_of,
        )

    def _load_fallback_whisper(self) -> None:
        self._load_whisper_model(self.fallback_model_id, model_role="Fallback")

    @property
    def active_faster_whisper_model_id(self) -> str:
        return self._active_faster_whisper_model_id

    def _resample_audio(self, audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        if source_rate <= 0 or source_rate == target_rate or audio.size == 0:
            return audio
        duration = float(audio.size) / float(source_rate)
        if duration <= 0.0:
            return audio
        target_size = max(1, int(round(duration * float(target_rate))))
        source_positions = np.linspace(0, audio.size - 1, num=audio.size, dtype=np.float32)
        target_positions = np.linspace(0, audio.size - 1, num=target_size, dtype=np.float32)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def _trim_non_speech_light_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0 or sample_rate <= 0:
            return audio

        frame = max(1, int(sample_rate * LIGHT_VAD_FRAME_SECONDS))
        hop = max(1, frame // 2)
        if audio.size < frame:
            return audio

        energies: list[float] = []
        starts: list[int] = []
        for start in range(0, audio.size - frame + 1, hop):
            window = audio[start : start + frame]
            energy = float(np.sqrt(np.mean(np.square(window))))
            energies.append(energy)
            starts.append(start)

        voiced = [i for i, e in enumerate(energies) if e >= LIGHT_VAD_ENERGY_THRESHOLD]
        if not voiced:
            return np.empty(0, dtype=np.float32)

        pad = int(sample_rate * LIGHT_VAD_PAD_SECONDS)
        start = max(0, starts[voiced[0]] - pad)
        end = min(audio.size, starts[voiced[-1]] + frame + pad)
        trimmed = audio[start:end]

        min_samples = int(sample_rate * LIGHT_VAD_MIN_SPEECH_SECONDS)
        if trimmed.size < min_samples:
            return np.empty(0, dtype=np.float32)
        return trimmed

    def _transcribe_fallback_whisper(
        self,
        audio_f32: np.ndarray,
        prompt_text: str,
        final_pass: bool,
    ) -> str:
        assert self._fallback_whisper is not None
        beam_size = self.finalize_beam_size if final_pass else self.realtime_beam_size
        best_of = self.finalize_best_of if final_pass else self.realtime_best_of
        condition_on_previous_text = (
            self.finalize_condition_on_previous_text if final_pass else self.realtime_condition_on_previous_text
        )
        audio_seconds = float(audio_f32.size) / float(TARGET_SAMPLE_RATE)
        use_vad_filter = audio_seconds >= VAD_FILTER_MIN_AUDIO_SECONDS
        kwargs: dict[str, Any] = {
            "language": "ja",
            "vad_filter": use_vad_filter,
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": 0.0,
            "no_speech_threshold": self.whisper_no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
        }
        if prompt_text and condition_on_previous_text:
            kwargs["initial_prompt"] = prompt_text
        segments, _ = self._fallback_whisper.transcribe(audio_f32, **kwargs)
        return "".join(seg.text for seg in segments).strip()

    def _transcribe_sapi(self, audio_f32: np.ndarray, sample_rate: int) -> str:
        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(self.cache_dir)) as temp_wav:
            wav_path = Path(temp_wav.name)

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())

            escaped_wav_path = str(wav_path.resolve()).replace("'", "''")
            command = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "$ErrorActionPreference='Stop'; "
                "Add-Type -AssemblyName System.Speech; "
                "$culture = New-Object System.Globalization.CultureInfo('ja-JP'); "
                "$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture); "
                "$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
                f"$engine.SetInputToWaveFile('{escaped_wav_path}'); "
                "$result = $engine.Recognize(); "
                "if ($result -ne $null) { "
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Write-Output $result.Text "
                "}",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=20,
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(stderr or f"powershell exit code={completed.returncode}")
            return completed.stdout.strip()
        finally:
            wav_path.unlink(missing_ok=True)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt_text: str = "",
        final_pass: bool = False,
    ) -> str:
        if audio.size == 0:
            return ""

        audio_f32 = np.clip(audio.astype(np.float32), -1.0, 1.0)
        normalized_sample_rate = int(sample_rate)
        if normalized_sample_rate != TARGET_SAMPLE_RATE:
            audio_f32 = self._resample_audio(audio_f32, normalized_sample_rate, TARGET_SAMPLE_RATE)
            normalized_sample_rate = TARGET_SAMPLE_RATE

        if self.backend in {"transformers", "faster-whisper"} and self.use_light_vad_for_whisper:
            trimmed_audio = self._trim_non_speech_light_vad(audio_f32, normalized_sample_rate)
            if trimmed_audio.size == 0:
                return ""
            audio_f32 = trimmed_audio
        elif self.backend == "transformers":
            trimmed_audio = self._trim_non_speech_light_vad(audio_f32, normalized_sample_rate)
            if trimmed_audio.size > 0:
                audio_f32 = trimmed_audio

        # Skip near-silent chunks to avoid common Whisper hallucinations
        rms = float(np.sqrt(np.mean(np.square(audio_f32))))
        min_rms = self.finalize_min_rms if final_pass else self.realtime_min_rms
        if rms < min_rms:
            return ""

        if self.backend == "sapi":
            try:
                return self._transcribe_sapi(audio_f32, normalized_sample_rate)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("SAPI transcribe failed (%s). Switching fallback.", exc)
                self._load_fallback_whisper()

        if self.backend == "sherpa-onnx" and self._sherpa_onnx_recognizer is not None:
            try:
                stream = self._sherpa_onnx_recognizer.create_stream()
                stream.accept_waveform(normalized_sample_rate, audio_f32)
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
                    {"array": audio_f32, "sampling_rate": normalized_sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "ja",
                        "temperature": 0.0,
                        "condition_on_prev_tokens": final_pass,
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

        return self._transcribe_fallback_whisper(
            audio_f32,
            prompt_text=prompt_text,
            final_pass=final_pass,
        )
