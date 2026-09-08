import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NaturalMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class NaturalMotionConfig:
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    output_fps: int = 25
    crf: int = 18
    minimum_front_ratio: float = 0.60


@dataclass(frozen=True)
class NaturalMotionWindow:
    source_video_path: Path
    start_seconds: float
    duration_seconds: float
    score: float
    front_frame_ratio: float
    confidence: str


@dataclass(frozen=True)
class NaturalMotionClipResult:
    source_video_path: Path
    clip_path: Path
    start_seconds: float
    duration_seconds: float
    score: float
    front_frame_ratio: float
    confidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceVideoPath": str(self.source_video_path),
            "clipPath": str(self.clip_path),
            "startSeconds": round(self.start_seconds, 3),
            "durationSeconds": round(self.duration_seconds, 3),
            "score": round(self.score, 2),
            "frontFrameRatio": round(self.front_frame_ratio, 3),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NaturalIdleClipResult:
    source_video_path: Path
    output_path: Path
    start_seconds: float
    segment_duration_seconds: float
    output_duration_seconds: float
    front_frame_ratio: float
    confidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceVideoPath": str(self.source_video_path),
            "outputPath": str(self.output_path),
            "startSeconds": round(self.start_seconds, 3),
            "segmentDurationSeconds": round(self.segment_duration_seconds, 3),
            "outputDurationSeconds": round(self.output_duration_seconds, 3),
            "frontFrameRatio": round(self.front_frame_ratio, 3),
            "confidence": self.confidence,
        }


def prepare_natural_motion_clip(
    manifest_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    config: NaturalMotionConfig | None = None,
) -> NaturalMotionClipResult:
    motion_config = config or NaturalMotionConfig()
    audio_duration = probe_media_duration(
        audio_path,
        ffprobe_binary=motion_config.ffprobe_binary,
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    window = select_natural_motion_window(
        data.get("videos", []),
        target_duration_seconds=audio_duration,
        minimum_front_ratio=motion_config.minimum_front_ratio,
    )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        motion_config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{window.start_seconds:.3f}",
        "-i",
        str(window.source_video_path),
        "-t",
        f"{audio_duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={motion_config.output_fps}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(motion_config.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise NaturalMotionError(
            f"ffmpeg executable not found: {motion_config.ffmpeg_binary}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffmpeg failed").strip()
        raise NaturalMotionError(f"Natural motion clip creation failed: {detail}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise NaturalMotionError(f"Natural motion clip was not created: {output_path}")

    return NaturalMotionClipResult(
        source_video_path=window.source_video_path,
        clip_path=output_path,
        start_seconds=window.start_seconds,
        duration_seconds=audio_duration,
        score=window.score,
        front_frame_ratio=window.front_frame_ratio,
        confidence=window.confidence,
    )


def prepare_natural_idle_clip(
    manifest_path: Path,
    output_path: Path,
    *,
    output_duration_seconds: float = 30.0,
    segment_duration_seconds: float = 2.5,
    config: NaturalMotionConfig | None = None,
) -> NaturalIdleClipResult:
    if output_duration_seconds <= 0 or segment_duration_seconds <= 0:
        raise NaturalMotionError("Idle clip durations must be greater than zero.")

    motion_config = config or NaturalMotionConfig()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    window = select_natural_motion_window(
        data.get("videos", []),
        target_duration_seconds=segment_duration_seconds,
        minimum_front_ratio=motion_config.minimum_front_ratio,
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_path = output_path.with_name(f".{output_path.stem}-cycle.mp4")

    cycle_command = [
        motion_config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{window.start_seconds:.3f}",
        "-i",
        str(window.source_video_path),
        "-t",
        f"{segment_duration_seconds:.3f}",
        "-an",
        "-filter_complex",
        (
            f"[0:v]fps={motion_config.output_fps},setpts=PTS-STARTPTS,"
            "split=2[forward][reverse_input];"
            "[reverse_input]reverse,setpts=PTS-STARTPTS[backward];"
            "[forward][backward]concat=n=2:v=1:a=0[video]"
        ),
        "-map",
        "[video]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        str(cycle_path),
    ]
    loop_command = [
        motion_config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(cycle_path),
        "-t",
        f"{output_duration_seconds:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(motion_config.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        _run_ffmpeg(cycle_command, motion_config.ffmpeg_binary)
        _run_ffmpeg(loop_command, motion_config.ffmpeg_binary)
    finally:
        cycle_path.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise NaturalMotionError(f"Natural idle clip was not created: {output_path}")
    return NaturalIdleClipResult(
        source_video_path=window.source_video_path,
        output_path=output_path,
        start_seconds=window.start_seconds,
        segment_duration_seconds=segment_duration_seconds,
        output_duration_seconds=output_duration_seconds,
        front_frame_ratio=window.front_frame_ratio,
        confidence=window.confidence,
    )


def select_natural_motion_window(
    videos: list[dict[str, Any]],
    *,
    target_duration_seconds: float,
    minimum_front_ratio: float = 0.60,
) -> NaturalMotionWindow:
    if target_duration_seconds <= 0:
        raise NaturalMotionError("Target duration must be greater than zero.")

    candidates: list[NaturalMotionWindow] = []
    for video in videos:
        local_path_value = video.get("localPath")
        metadata = video.get("metadata") or {}
        selection = video.get("frameSelection") or {}
        frames = selection.get("frames") or []
        try:
            source_duration = float(metadata["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if not local_path_value or not frames or source_duration < target_duration_seconds:
            continue

        sample_interval = source_duration / len(frames)
        window_frame_count = max(
            2,
            min(len(frames), math.ceil(target_duration_seconds / sample_interval)),
        )
        for start_index in range(len(frames) - window_frame_count + 1):
            window_frames = frames[start_index : start_index + window_frame_count]
            front_count = sum(frame.get("view") == "front" for frame in window_frames)
            front_ratio = front_count / len(window_frames)
            mean_quality = sum(
                float(frame.get("qualityScore", 0.0)) for frame in window_frames
            ) / len(window_frames)
            penalty = sum(_frame_penalty(frame) for frame in window_frames) / len(
                window_frames
            )
            score = mean_quality + 25.0 * front_ratio - penalty
            start_seconds = min(
                start_index * sample_interval,
                source_duration - target_duration_seconds,
            )
            candidates.append(
                NaturalMotionWindow(
                    source_video_path=Path(local_path_value),
                    start_seconds=max(0.0, start_seconds),
                    duration_seconds=target_duration_seconds,
                    score=round(score, 2),
                    front_frame_ratio=front_ratio,
                    confidence=_window_confidence(
                        front_ratio,
                        mean_quality,
                        minimum_front_ratio,
                    ),
                )
            )

    if not candidates:
        raise NaturalMotionError(
            "Manifest has no source video long enough for the preview audio."
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate.front_frame_ratio >= minimum_front_ratio,
            candidate.score,
        ),
    )


def probe_media_duration(path: Path, *, ffprobe_binary: str = "ffprobe") -> float:
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise NaturalMotionError(f"ffprobe executable not found: {ffprobe_binary}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffprobe failed").strip()
        raise NaturalMotionError(f"Unable to inspect media duration: {detail}") from exc

    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NaturalMotionError(f"Media duration is missing: {path}") from exc
    if duration <= 0:
        raise NaturalMotionError(f"Media duration must be greater than zero: {path}")
    return duration


def _run_ffmpeg(command: list[str], ffmpeg_binary: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise NaturalMotionError(f"ffmpeg executable not found: {ffmpeg_binary}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffmpeg failed").strip()
        raise NaturalMotionError(f"Natural idle clip creation failed: {detail}") from exc


def _frame_penalty(frame: dict[str, Any]) -> float:
    reasons = set(frame.get("rejectionReasons") or [])
    penalties = {
        "face_not_detected": 45.0,
        "multiple_faces_detected": 35.0,
        "face_too_small": 20.0,
        "face_too_close": 15.0,
        "face_off_center": 15.0,
        "too_blurry": 8.0,
        "too_dark": 12.0,
        "too_bright": 12.0,
        "low_contrast": 8.0,
    }
    return sum(penalties.get(reason, 5.0) for reason in reasons)


def _window_confidence(
    front_ratio: float,
    mean_quality: float,
    minimum_front_ratio: float,
) -> str:
    if front_ratio >= 0.8 and mean_quality >= 65.0:
        return "high"
    if front_ratio >= minimum_front_ratio and mean_quality >= 50.0:
        return "medium"
    return "low"
