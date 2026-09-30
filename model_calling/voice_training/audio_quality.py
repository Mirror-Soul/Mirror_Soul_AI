import math
import os
import subprocess
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VoiceTrainingAudioSample:
    filename: str
    content: bytes
    content_type: str


@dataclass(frozen=True)
class VoiceTrainingAudioMetrics:
    duration_seconds: float
    rms_dbfs: float
    peak_dbfs: float
    silence_ratio: float
    clipping_ratio: float


@dataclass(frozen=True)
class ValidatedVoiceTrainingAudio:
    sample_number: int
    filename: str
    content: bytes
    content_type: str
    metrics: VoiceTrainingAudioMetrics


@dataclass(frozen=True)
class RejectedVoiceTrainingAudio:
    sample_number: int
    reason_codes: tuple[str, ...]
    metrics: VoiceTrainingAudioMetrics | None = None


@dataclass(frozen=True)
class VoiceTrainingAudioBatch:
    accepted: tuple[ValidatedVoiceTrainingAudio, ...]
    rejected: tuple[RejectedVoiceTrainingAudio, ...]
    total_duration_seconds: float


@dataclass(frozen=True)
class VoiceTrainingAudioQualityConfig:
    ffmpeg_binary: str = "ffmpeg"
    sample_rate: int = 16000
    min_sample_duration_seconds: float = 0.8
    max_sample_duration_seconds: float = 120.0
    min_batch_duration_seconds: float = 8.0
    min_accepted_samples: int = 3
    min_rms_dbfs: float = -48.0
    silence_threshold_dbfs: float = -45.0
    max_silence_ratio: float = 0.80
    max_clipping_ratio: float = 0.03
    max_file_bytes: int = 25 * 1024 * 1024
    normalize_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "VoiceTrainingAudioQualityConfig":
        return cls(
            ffmpeg_binary=(
                os.getenv("VOICE_TRAINING_FFMPEG_BINARY")
                or os.getenv("FFMPEG_BINARY")
                or os.getenv("FFMPEG_BIN")
                or "ffmpeg"
            ),
            sample_rate=_env_int("VOICE_TRAINING_AUDIO_SAMPLE_RATE", 16000),
            min_sample_duration_seconds=_env_float(
                "VOICE_TRAINING_MIN_SAMPLE_DURATION_SECONDS", 0.8
            ),
            max_sample_duration_seconds=_env_float(
                "VOICE_TRAINING_MAX_SAMPLE_DURATION_SECONDS", 120.0
            ),
            min_batch_duration_seconds=_env_float(
                "VOICE_TRAINING_MIN_BATCH_DURATION_SECONDS", 8.0
            ),
            min_accepted_samples=_env_int(
                "VOICE_TRAINING_MIN_ACCEPTED_SAMPLES", 3
            ),
            min_rms_dbfs=_env_float("VOICE_TRAINING_MIN_RMS_DBFS", -48.0),
            silence_threshold_dbfs=_env_float(
                "VOICE_TRAINING_SILENCE_THRESHOLD_DBFS", -45.0
            ),
            max_silence_ratio=_env_float(
                "VOICE_TRAINING_MAX_SILENCE_RATIO", 0.80
            ),
            max_clipping_ratio=_env_float(
                "VOICE_TRAINING_MAX_CLIPPING_RATIO", 0.03
            ),
            max_file_bytes=_env_int(
                "VOICE_TRAINING_MAX_FILE_BYTES", 25 * 1024 * 1024
            ),
            normalize_timeout_seconds=_env_float(
                "VOICE_TRAINING_NORMALIZE_TIMEOUT_SECONDS", 60.0
            ),
        )


class VoiceTrainingInputQualityError(Exception):
    def __init__(
        self,
        code: str,
        batch: VoiceTrainingAudioBatch,
    ) -> None:
        self.code = code
        self.batch = batch
        super().__init__(
            "voice_input_quality_failed: "
            f"code={code} accepted={len(batch.accepted)} "
            f"rejected={len(batch.rejected)} "
            f"duration={batch.total_duration_seconds:.2f}s"
        )


class _AudioNormalizationError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def normalize_and_validate_voice_samples(
    samples: Sequence[VoiceTrainingAudioSample],
    *,
    config: VoiceTrainingAudioQualityConfig | None = None,
) -> VoiceTrainingAudioBatch:
    quality_config = config or VoiceTrainingAudioQualityConfig.from_env()
    accepted: list[ValidatedVoiceTrainingAudio] = []
    rejected: list[RejectedVoiceTrainingAudio] = []

    with tempfile.TemporaryDirectory(prefix="mirror-soul-voice-input-") as temp_dir:
        workspace = Path(temp_dir)
        for sample_number, sample in enumerate(samples, start=1):
            try:
                normalized_path = _normalize_sample(
                    sample,
                    sample_number=sample_number,
                    workspace=workspace,
                    config=quality_config,
                )
                metrics = analyze_normalized_wav(
                    normalized_path,
                    silence_threshold_dbfs=quality_config.silence_threshold_dbfs,
                )
                reason_codes = _quality_rejection_reasons(metrics, quality_config)
                if reason_codes:
                    rejected.append(
                        RejectedVoiceTrainingAudio(
                            sample_number,
                            reason_codes,
                            metrics,
                        )
                    )
                    continue

                accepted.append(
                    ValidatedVoiceTrainingAudio(
                        sample_number=sample_number,
                        filename=f"voice-sample-{sample_number:02d}.wav",
                        content=normalized_path.read_bytes(),
                        content_type="audio/wav",
                        metrics=metrics,
                    )
                )
            except _AudioNormalizationError as exc:
                rejected.append(
                    RejectedVoiceTrainingAudio(sample_number, (exc.code,))
                )

    batch = VoiceTrainingAudioBatch(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        total_duration_seconds=round(
            sum(sample.metrics.duration_seconds for sample in accepted), 2
        ),
    )
    if len(batch.accepted) < quality_config.min_accepted_samples:
        raise VoiceTrainingInputQualityError("insufficient_valid_samples", batch)
    if batch.total_duration_seconds < quality_config.min_batch_duration_seconds:
        raise VoiceTrainingInputQualityError("insufficient_valid_duration", batch)
    return batch


def analyze_normalized_wav(
    path: Path,
    *,
    silence_threshold_dbfs: float = -45.0,
) -> VoiceTrainingAudioMetrics:
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() <= 0
            ):
                raise _AudioNormalizationError("invalid_normalized_format")
            sample_rate = audio.getframerate()
            pcm = array("h")
            pcm.frombytes(audio.readframes(audio.getnframes()))
    except (EOFError, wave.Error, OSError) as exc:
        raise _AudioNormalizationError("invalid_normalized_audio") from exc

    if not pcm:
        raise _AudioNormalizationError("empty_normalized_audio")

    full_scale = 32768.0
    sample_count = len(pcm)
    squares = sum(float(value) * float(value) for value in pcm)
    rms = math.sqrt(squares / sample_count)
    peak = max(abs(value) for value in pcm)
    clipping_threshold = int(full_scale * 0.999)
    clipping_ratio = sum(abs(value) >= clipping_threshold for value in pcm) / sample_count

    frame_size = max(1, int(sample_rate * 0.02))
    silent_frames = 0
    frame_count = 0
    silence_amplitude = full_scale * (10.0 ** (silence_threshold_dbfs / 20.0))
    for offset in range(0, sample_count, frame_size):
        frame = pcm[offset : offset + frame_size]
        if not frame:
            continue
        frame_rms = math.sqrt(
            sum(float(value) * float(value) for value in frame) / len(frame)
        )
        frame_count += 1
        if frame_rms < silence_amplitude:
            silent_frames += 1

    return VoiceTrainingAudioMetrics(
        duration_seconds=round(sample_count / sample_rate, 3),
        rms_dbfs=round(_amplitude_to_dbfs(rms, full_scale), 2),
        peak_dbfs=round(_amplitude_to_dbfs(peak, full_scale), 2),
        silence_ratio=round(silent_frames / max(1, frame_count), 4),
        clipping_ratio=round(clipping_ratio, 4),
    )


def _normalize_sample(
    sample: VoiceTrainingAudioSample,
    *,
    sample_number: int,
    workspace: Path,
    config: VoiceTrainingAudioQualityConfig,
) -> Path:
    if not sample.content:
        raise _AudioNormalizationError("empty_input")
    if len(sample.content) > config.max_file_bytes:
        raise _AudioNormalizationError("file_too_large")

    source_path = workspace / f"source-{sample_number:02d}{_safe_suffix(sample)}"
    output_path = workspace / f"normalized-{sample_number:02d}.wav"
    source_path.write_bytes(sample.content)
    command = [
        config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.normalize_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise _AudioNormalizationError("ffmpeg_not_found") from exc
    except subprocess.TimeoutExpired as exc:
        raise _AudioNormalizationError("normalization_timeout") from exc
    except subprocess.CalledProcessError as exc:
        raise _AudioNormalizationError("decode_failed") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise _AudioNormalizationError("normalization_empty_output")
    return output_path


def _quality_rejection_reasons(
    metrics: VoiceTrainingAudioMetrics,
    config: VoiceTrainingAudioQualityConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.duration_seconds < config.min_sample_duration_seconds:
        reasons.append("sample_too_short")
    if metrics.duration_seconds > config.max_sample_duration_seconds:
        reasons.append("sample_too_long")
    if metrics.rms_dbfs < config.min_rms_dbfs:
        reasons.append("audio_too_quiet")
    if metrics.silence_ratio > config.max_silence_ratio:
        reasons.append("too_much_silence")
    if metrics.clipping_ratio > config.max_clipping_ratio:
        reasons.append("audio_clipping")
    return tuple(reasons)


def _safe_suffix(sample: VoiceTrainingAudioSample) -> str:
    suffix = Path(sample.filename).suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".aac"}:
        return suffix
    content_type = sample.content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    if "mp4" in content_type or "m4a" in content_type:
        return ".m4a"
    if "webm" in content_type:
        return ".webm"
    if "ogg" in content_type:
        return ".ogg"
    if "aac" in content_type:
        return ".aac"
    return ".wav"


def _amplitude_to_dbfs(amplitude: float, full_scale: float) -> float:
    if amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(min(amplitude / full_scale, 1.0))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default
