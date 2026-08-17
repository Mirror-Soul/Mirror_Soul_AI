import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class FaceVideoProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceVideoMetadata:
    duration_seconds: float
    width: int
    height: int
    frame_rate: float
    codec_name: str
    format_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaceVideoPreprocessResult:
    source_path: Path
    metadata: FaceVideoMetadata
    frame_paths: list[Path]


def preprocess_face_video(
    video_path: Path,
    frames_dir: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    ffmpeg_binary: str = "ffmpeg",
    extraction_fps: float = 2.0,
    max_frames: int = 120,
    min_resolution: int = 256,
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 120.0,
) -> FaceVideoPreprocessResult:
    metadata = probe_face_video(video_path, ffprobe_binary=ffprobe_binary)
    validate_face_video_metadata(
        metadata,
        min_resolution=min_resolution,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )
    frame_paths = extract_face_frames(
        video_path,
        frames_dir,
        ffmpeg_binary=ffmpeg_binary,
        extraction_fps=extraction_fps,
        max_frames=max_frames,
    )
    return FaceVideoPreprocessResult(
        source_path=video_path,
        metadata=metadata,
        frame_paths=frame_paths,
    )


def probe_face_video(
    video_path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
) -> FaceVideoMetadata:
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FaceVideoProcessingError(
            f"ffprobe executable not found: {ffprobe_binary}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffprobe failed").strip()
        raise FaceVideoProcessingError(f"Unable to inspect video: {detail}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FaceVideoProcessingError("ffprobe returned invalid JSON.") from exc
    return metadata_from_ffprobe_payload(payload)


def metadata_from_ffprobe_payload(payload: dict[str, Any]) -> FaceVideoMetadata:
    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise FaceVideoProcessingError("No video stream found.")

    format_info = payload.get("format") or {}
    duration_value = video_stream.get("duration") or format_info.get("duration")
    try:
        duration_seconds = float(duration_value)
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FaceVideoProcessingError(
            "Video duration or resolution metadata is missing."
        ) from exc

    return FaceVideoMetadata(
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        frame_rate=_parse_frame_rate(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        ),
        codec_name=str(video_stream.get("codec_name") or "unknown"),
        format_name=str(format_info.get("format_name") or "unknown"),
    )


def validate_face_video_metadata(
    metadata: FaceVideoMetadata,
    *,
    min_resolution: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> None:
    if not min_duration_seconds <= metadata.duration_seconds <= max_duration_seconds:
        raise FaceVideoProcessingError(
            "Video duration must be between "
            f"{min_duration_seconds:g} and {max_duration_seconds:g} seconds; "
            f"received {metadata.duration_seconds:.2f}."
        )
    if min(metadata.width, metadata.height) < min_resolution:
        raise FaceVideoProcessingError(
            f"Video resolution must be at least {min_resolution}px on both axes; "
            f"received {metadata.width}x{metadata.height}."
        )
    if metadata.frame_rate <= 0:
        raise FaceVideoProcessingError("Video frame rate must be greater than zero.")


def extract_face_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    extraction_fps: float = 2.0,
    max_frames: int = 120,
) -> list[Path]:
    if extraction_fps <= 0 or max_frames <= 0:
        raise FaceVideoProcessingError(
            "Frame extraction fps and max_frames must be greater than zero."
        )

    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame-%05d.jpg"
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={extraction_fps:g}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FaceVideoProcessingError(
            f"ffmpeg executable not found: {ffmpeg_binary}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffmpeg failed").strip()
        raise FaceVideoProcessingError(f"Frame extraction failed: {detail}") from exc

    frame_paths = sorted(frames_dir.glob("frame-*.jpg"))
    if not frame_paths:
        raise FaceVideoProcessingError("Frame extraction produced no images.")
    return frame_paths


def _parse_frame_rate(value: Any) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    text = str(value)
    if "/" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    numerator, denominator = text.split("/", 1)
    try:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    except ValueError:
        return 0.0

