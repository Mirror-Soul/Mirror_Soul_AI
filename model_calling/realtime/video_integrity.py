from __future__ import annotations

import io
import os
from dataclasses import dataclass

import av


@dataclass(frozen=True)
class VideoIntegrityConfig:
    enabled: bool = True
    min_frames: int = 2
    min_duration_seconds: float = 0.20
    min_frame_rate: float = 10.0
    max_frame_rate: float = 60.0
    max_duration_delta_seconds: float = 1.25
    max_duration_delta_ratio: float = 0.35

    @classmethod
    def from_env(cls) -> "VideoIntegrityConfig":
        config = cls(
            enabled=_env_bool("DITTO_CALL_VIDEO_VALIDATION_ENABLED", True),
            min_frames=_env_int("DITTO_CALL_VIDEO_MIN_FRAMES", 2),
            min_duration_seconds=_env_float(
                "DITTO_CALL_VIDEO_MIN_DURATION_SECONDS", 0.20
            ),
            min_frame_rate=_env_float("DITTO_CALL_VIDEO_MIN_FPS", 10.0),
            max_frame_rate=_env_float("DITTO_CALL_VIDEO_MAX_FPS", 60.0),
            max_duration_delta_seconds=_env_float(
                "DITTO_CALL_VIDEO_MAX_DURATION_DELTA_SECONDS", 1.25
            ),
            max_duration_delta_ratio=_env_float(
                "DITTO_CALL_VIDEO_MAX_DURATION_DELTA_RATIO", 0.35
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.min_frames <= 0:
            raise ValueError("DITTO_CALL_VIDEO_MIN_FRAMES must be positive.")
        if self.min_duration_seconds <= 0:
            raise ValueError(
                "DITTO_CALL_VIDEO_MIN_DURATION_SECONDS must be positive."
            )
        if self.min_frame_rate <= 0 or self.max_frame_rate < self.min_frame_rate:
            raise ValueError("Ditto video FPS limits are invalid.")
        if (
            self.max_duration_delta_seconds < 0
            or self.max_duration_delta_ratio < 0
        ):
            raise ValueError("Ditto video duration tolerances must not be negative.")


@dataclass(frozen=True)
class VideoIntegrityResult:
    frame_count: int
    duration_seconds: float
    expected_audio_duration_seconds: float
    frame_rate: float
    width: int
    height: int
    duration_delta_seconds: float


class VideoIntegrityError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def validate_rendered_video(
    video_bytes: bytes,
    expected_audio_bytes: bytes,
    *,
    config: VideoIntegrityConfig,
) -> VideoIntegrityResult:
    if not config.enabled:
        return VideoIntegrityResult(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    if not video_bytes:
        raise VideoIntegrityError(
            "Rendered video is empty.",
            code="DITTO_VIDEO_EMPTY",
        )

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                raise VideoIntegrityError(
                    "Rendered MP4 does not contain a video stream.",
                    code="DITTO_VIDEO_STREAM_MISSING",
                )
            stream = container.streams.video[0]
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
            frame_rate = _frame_rate(stream)
            frame_count = 0
            first_pts_seconds: float | None = None
            last_pts_seconds: float | None = None
            for frame in container.decode(video=0):
                frame_count += 1
                if frame.pts is not None and frame.time_base is not None:
                    pts_seconds = float(frame.pts * frame.time_base)
                    if first_pts_seconds is None:
                        first_pts_seconds = pts_seconds
                    last_pts_seconds = pts_seconds
            duration_seconds = _video_duration(
                container,
                stream,
                frame_count=frame_count,
                frame_rate=frame_rate,
                first_pts_seconds=first_pts_seconds,
                last_pts_seconds=last_pts_seconds,
            )
    except VideoIntegrityError:
        raise
    except Exception as exc:
        raise VideoIntegrityError(
            "Rendered MP4 could not be decoded.",
            code="DITTO_VIDEO_DECODE_FAILED",
        ) from exc

    if frame_count < config.min_frames:
        raise VideoIntegrityError(
            f"Rendered MP4 has too few frames: {frame_count}.",
            code="DITTO_VIDEO_TOO_FEW_FRAMES",
        )
    if duration_seconds < config.min_duration_seconds:
        raise VideoIntegrityError(
            f"Rendered MP4 is too short: {duration_seconds:.3f}s.",
            code="DITTO_VIDEO_TOO_SHORT",
        )
    if width <= 0 or height <= 0:
        raise VideoIntegrityError(
            "Rendered MP4 has invalid dimensions.",
            code="DITTO_VIDEO_DIMENSIONS_INVALID",
        )
    if not config.min_frame_rate <= frame_rate <= config.max_frame_rate:
        raise VideoIntegrityError(
            f"Rendered MP4 frame rate is invalid: {frame_rate:.3f}.",
            code="DITTO_VIDEO_FPS_INVALID",
        )

    expected_duration = _audio_duration(expected_audio_bytes)
    duration_delta = abs(duration_seconds - expected_duration)
    allowed_delta = max(
        config.max_duration_delta_seconds,
        expected_duration * config.max_duration_delta_ratio,
    )
    if duration_delta > allowed_delta:
        raise VideoIntegrityError(
            "Rendered MP4 duration does not match reply audio: "
            f"video={duration_seconds:.3f}s audio={expected_duration:.3f}s "
            f"delta={duration_delta:.3f}s allowed={allowed_delta:.3f}s.",
            code="DITTO_VIDEO_DURATION_MISMATCH",
        )

    return VideoIntegrityResult(
        frame_count=frame_count,
        duration_seconds=round(duration_seconds, 3),
        expected_audio_duration_seconds=round(expected_duration, 3),
        frame_rate=round(frame_rate, 3),
        width=width,
        height=height,
        duration_delta_seconds=round(duration_delta, 3),
    )


def _audio_duration(audio_bytes: bytes) -> float:
    if not audio_bytes:
        raise VideoIntegrityError(
            "Reply audio is empty.",
            code="DITTO_REPLY_AUDIO_INVALID",
        )
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            if not container.streams.audio:
                raise VideoIntegrityError(
                    "Reply audio does not contain an audio stream.",
                    code="DITTO_REPLY_AUDIO_INVALID",
                )
            stream = container.streams.audio[0]
            duration = _stream_duration(stream)
            if duration <= 0 and container.duration:
                duration = float(container.duration / av.time_base)
            if duration <= 0:
                samples = 0
                sample_rate = int(stream.codec_context.sample_rate or 0)
                for frame in container.decode(audio=0):
                    samples += int(frame.samples)
                    sample_rate = sample_rate or int(frame.sample_rate or 0)
                if sample_rate > 0:
                    duration = samples / sample_rate
    except VideoIntegrityError:
        raise
    except Exception as exc:
        raise VideoIntegrityError(
            "Reply audio could not be inspected.",
            code="DITTO_REPLY_AUDIO_INVALID",
        ) from exc
    if duration <= 0:
        raise VideoIntegrityError(
            "Reply audio duration is unavailable.",
            code="DITTO_REPLY_AUDIO_INVALID",
        )
    return duration


def _video_duration(
    container,
    stream,
    *,
    frame_count: int,
    frame_rate: float,
    first_pts_seconds: float | None,
    last_pts_seconds: float | None,
) -> float:
    duration = _stream_duration(stream)
    if duration <= 0 and container.duration:
        duration = float(container.duration / av.time_base)
    if duration <= 0 and frame_rate > 0:
        duration = frame_count / frame_rate
    if (
        duration <= 0
        and first_pts_seconds is not None
        and last_pts_seconds is not None
        and frame_rate > 0
    ):
        duration = last_pts_seconds - first_pts_seconds + (1.0 / frame_rate)
    return duration


def _stream_duration(stream) -> float:
    if stream.duration is None or stream.time_base is None:
        return 0.0
    return float(stream.duration * stream.time_base)


def _frame_rate(stream) -> float:
    rate = stream.average_rate or stream.base_rate or stream.guessed_rate
    return float(rate) if rate else 0.0


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default
