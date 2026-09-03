import argparse
import json
import mimetypes
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from model_training.face_training.frame_analyzer import (
    FrameQualityConfig,
    FrameSelectionResult,
    analyze_face_frames,
)
from model_training.face_training.face_similarity import (
    FaceSimilarityResult,
    FaceSimilarityUnavailable,
    evaluate_face_similarity,
)
from model_training.face_training.message import (
    FaceTrainingMessage,
    FaceTrainingMessageError,
    parse_face_training_message,
)
from model_training.face_training.liveportrait_runner import (
    LivePortraitConfig,
    LivePortraitResult,
    run_liveportrait,
)
from model_training.face_training.video_processor import (
    FaceVideoPreprocessResult,
    preprocess_face_video,
)

load_dotenv()


class FaceTrainingWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadedFaceVideo:
    bucket: str
    object_key: str
    content_type: str
    local_path: Path
    size_bytes: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume Mirror Soul face profile jobs from SQS."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one available SQS message, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Download and preprocess one job without deleting its SQS message. "
            "Required until the LivePortrait completion pipeline is connected."
        ),
    )
    args = parser.parse_args()
    run_worker(once=args.once, dry_run=args.dry_run)


def run_worker(*, once: bool = False, dry_run: bool = False) -> None:
    if not (once and dry_run):
        raise FaceTrainingWorkerError(
            "The first worker version only supports --once --dry-run so that "
            "production SQS messages cannot be deleted prematurely."
        )

    queue_url = os.getenv("AWS_SQS_FACE_TRAINING_QUEUE_URL")
    if not queue_url:
        raise FaceTrainingWorkerError(
            "AWS_SQS_FACE_TRAINING_QUEUE_URL is not configured."
        )

    sqs_client = _boto3_client("sqs")
    s3_client = _boto3_client("s3")
    wait_seconds = _env_int("FACE_TRAINING_WAIT_SECONDS", 20)
    visibility_timeout = _env_int("FACE_TRAINING_VISIBILITY_TIMEOUT", 900)

    print("[FACE_TRAINING] worker started: mode=dry-run", flush=True)
    response = sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=wait_seconds,
        VisibilityTimeout=visibility_timeout,
    )
    messages = response.get("Messages", [])
    if not messages:
        print("[FACE_TRAINING] no message available", flush=True)
        return

    sqs_message = messages[0]
    try:
        _handle_sqs_message(s3_client, sqs_message)
    finally:
        receipt_handle = sqs_message.get("ReceiptHandle")
        if receipt_handle:
            sqs_client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=0,
            )
            print(
                "[FACE_TRAINING] dry-run message visibility restored",
                flush=True,
            )


def _handle_sqs_message(s3_client: Any, sqs_message: dict[str, Any]) -> None:
    try:
        message = parse_face_training_message(sqs_message.get("Body", ""))
    except FaceTrainingMessageError as exc:
        print(f"[FACE_TRAINING] invalid message retained: {exc}", flush=True)
        return

    try:
        manifest_path = _preprocess_face_training_message(s3_client, message)
    except Exception as exc:
        print(
            f"[FACE_TRAINING] dry-run failed: job_id={message.job_id} error={exc}",
            flush=True,
        )
        return

    print(
        "[FACE_TRAINING] dry-run completed; SQS message retained: "
        f"job_id={message.job_id} manifest={manifest_path}",
        flush=True,
    )


def _preprocess_face_training_message(
    s3_client: Any,
    message: FaceTrainingMessage,
) -> Path:
    workspace = _create_run_workspace(message)
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[FACE_TRAINING] preprocessing: "
        f"job_id={message.job_id} user_uuid={message.user_uuid} "
        f"clone_id={message.clone_id} files={len(message.object_keys)}",
        flush=True,
    )

    downloaded_videos = [
        _download_face_video(
            s3_client,
            bucket=message.bucket,
            object_key=object_key,
            destination=inputs_dir / f"{index:02d}-{_safe_filename(object_key)}",
        )
        for index, object_key in enumerate(message.object_keys, start=1)
    ]

    preprocess_results = []
    frame_selection_results = []
    for index, video in enumerate(downloaded_videos, start=1):
        result = preprocess_face_video(
            video.local_path,
            workspace / "frames" / f"video-{index:02d}",
            ffprobe_binary=os.getenv("FFPROBE_BINARY", "ffprobe"),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            extraction_fps=_env_float("FACE_TRAINING_EXTRACTION_FPS", 2.0),
            max_frames=_env_int("FACE_TRAINING_MAX_EXTRACTED_FRAMES", 120),
            min_resolution=_env_int("FACE_TRAINING_MIN_RESOLUTION", 256),
            min_duration_seconds=_env_float(
                "FACE_TRAINING_MIN_DURATION_SECONDS", 1.0
            ),
            max_duration_seconds=_env_float(
                "FACE_TRAINING_MAX_DURATION_SECONDS", 120.0
            ),
        )
        preprocess_results.append(result)
        selection = analyze_face_frames(
            result.frame_paths,
            config=_frame_quality_config(),
        )
        frame_selection_results.append(selection)
        print(
            "[FACE_TRAINING] video preprocessed: "
            f"path={video.local_path} duration={result.metadata.duration_seconds:.2f}s "
            f"resolution={result.metadata.width}x{result.metadata.height} "
            f"frames={len(result.frame_paths)}",
            flush=True,
        )
        print(
            "[FACE_TRAINING] frame analysis completed: "
            f"accepted={selection.accepted_count} "
            f"rejected={selection.rejected_count} "
            f"quality_gate={selection.quality_gate_passed} "
            f"source={selection.selected_source_path}",
            flush=True,
        )

    liveportrait_result = None
    face_similarity_result = None
    if _env_bool("FACE_TRAINING_RUN_LIVEPORTRAIT", False):
        source_path, driving_path = _select_liveportrait_inputs(
            downloaded_videos,
            frame_selection_results,
        )
        print(
            "[FACE_TRAINING] LivePortrait started: "
            f"source={source_path} driving={driving_path}",
            flush=True,
        )
        liveportrait_result = run_liveportrait(
            source_path,
            driving_path,
            workspace / "outputs" / "liveportrait",
            config=_liveportrait_config(),
        )
        print(
            "[FACE_TRAINING] LivePortrait completed: "
            f"output={liveportrait_result.output_path} "
            f"duration={liveportrait_result.duration_seconds:.2f}s",
            flush=True,
        )
        face_similarity_result = _evaluate_liveportrait_similarity(
            frame_selection_results,
            liveportrait_result,
        )

    manifest_path = workspace / "preprocess-manifest.json"
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                message,
                downloaded_videos,
                preprocess_results,
                frame_selection_results,
                liveportrait_result,
                face_similarity_result,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _download_face_video(
    s3_client: Any,
    *,
    bucket: str,
    object_key: str,
    destination: Path,
) -> DownloadedFaceVideo:
    response = s3_client.get_object(Bucket=bucket, Key=object_key)
    content_type = (
        response.get("ContentType")
        or mimetypes.guess_type(object_key)[0]
        or "application/octet-stream"
    )
    allowed_types = {"video/mp4", "video/quicktime", "video/webm"}
    if content_type.lower() not in allowed_types:
        raise FaceTrainingWorkerError(
            f"Unsupported face video content type: {content_type}"
        )

    max_size_bytes = _env_int(
        "FACE_TRAINING_MAX_FILE_SIZE_BYTES", 100 * 1024 * 1024
    )
    declared_size = response.get("ContentLength")
    if declared_size is not None and int(declared_size) > max_size_bytes:
        raise FaceTrainingWorkerError(
            f"Face video exceeds {max_size_bytes} bytes: s3://{bucket}/{object_key}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    body = response["Body"]
    size_bytes = 0
    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_size_bytes:
                    raise FaceTrainingWorkerError(
                        f"Face video exceeds {max_size_bytes} bytes while downloading."
                    )
                output_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()

    if size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise FaceTrainingWorkerError(
            f"Empty S3 object: s3://{bucket}/{object_key}"
        )

    return DownloadedFaceVideo(
        bucket=bucket,
        object_key=object_key,
        content_type=content_type,
        local_path=destination,
        size_bytes=size_bytes,
    )


def _create_run_workspace(message: FaceTrainingMessage) -> Path:
    base_dir = Path(
        os.getenv("FACE_TRAINING_WORKSPACE_DIR", "tmp/face_training")
    ).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    workspace = base_dir / message.user_uuid / f"job-{message.job_id}" / timestamp
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _build_manifest(
    message: FaceTrainingMessage,
    downloaded_videos: list[DownloadedFaceVideo],
    preprocess_results: list[FaceVideoPreprocessResult],
    frame_selection_results: list[FrameSelectionResult],
    liveportrait_result: LivePortraitResult | None = None,
    face_similarity_result: FaceSimilarityResult | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "message": asdict(message),
        "videos": [
            {
                "bucket": video.bucket,
                "objectKey": video.object_key,
                "contentType": video.content_type,
                "localPath": str(video.local_path),
                "sizeBytes": video.size_bytes,
                "metadata": result.metadata.to_dict(),
                "frames": [str(path) for path in result.frame_paths],
                "frameSelection": selection.to_dict(),
            }
            for video, result, selection in zip(
                downloaded_videos,
                preprocess_results,
                frame_selection_results,
            )
        ],
        "livePortrait": (
            liveportrait_result.to_dict()
            if liveportrait_result is not None
            else None
        ),
        "faceSimilarity": (
            face_similarity_result.to_dict()
            if face_similarity_result is not None
            else None
        ),
    }


def _evaluate_liveportrait_similarity(
    frame_selection_results: list[FrameSelectionResult],
    liveportrait_result: LivePortraitResult,
) -> FaceSimilarityResult | None:
    if not _env_bool("FACE_SIMILARITY_ENABLE", False):
        return None

    reference_paths = [
        frame.path
        for selection in frame_selection_results
        for frame in selection.frames
        if frame.accepted
    ]
    required = _env_bool("FACE_SIMILARITY_REQUIRED", False)
    try:
        result = evaluate_face_similarity(
            reference_images=reference_paths,
            generated_video=liveportrait_result.output_path,
            driving_video=liveportrait_result.driving_path,
        )
    except FaceSimilarityUnavailable as exc:
        if required:
            raise FaceTrainingWorkerError(
                f"Face similarity evaluation is required but unavailable: {exc}"
            ) from exc
        print(
            f"[FACE_SIMILARITY] skipped: reason={exc}",
            flush=True,
        )
        return None
    except Exception as exc:
        if required:
            raise FaceTrainingWorkerError(
                f"Face similarity evaluation failed: {exc}"
            ) from exc
        print(
            f"[FACE_SIMILARITY] failed but face output was retained: error={exc}",
            flush=True,
        )
        return None

    print(
        "[FACE_SIMILARITY] completed: "
        f"score={result.score:.2f} identity={result.identity_score:.2f} "
        f"render={result.render_quality_score:.2f} "
        f"confidence={result.confidence} calibrated={result.calibrated}",
        flush=True,
    )
    return result


def _select_liveportrait_inputs(
    downloaded_videos: list[DownloadedFaceVideo],
    frame_selection_results: list[FrameSelectionResult],
) -> tuple[Path, Path]:
    candidates = []
    for video, selection in zip(downloaded_videos, frame_selection_results):
        if not selection.quality_gate_passed or selection.selected_source_path is None:
            continue
        selected_analysis = next(
            (
                frame
                for frame in selection.frames
                if frame.path == selection.selected_source_path
            ),
            None,
        )
        if selected_analysis is not None:
            candidates.append(
                (
                    selected_analysis.quality_score,
                    selection.selected_source_path,
                    video.local_path,
                )
            )

    if not candidates:
        raise FaceTrainingWorkerError(
            "No face video passed the quality gate for LivePortrait."
        )

    _, source_path, driving_path = max(candidates, key=lambda item: item[0])
    return source_path, driving_path


def _liveportrait_config() -> LivePortraitConfig:
    return LivePortraitConfig(
        repository_dir=Path(
            os.getenv(
                "FACE_TRAINING_LIVEPORTRAIT_REPO_DIR",
                "/workspace/mirror-soul-face/liveportrait",
            )
        ),
        python_binary=Path(
            os.getenv("FACE_TRAINING_LIVEPORTRAIT_PYTHON", sys.executable)
        ),
        timeout_seconds=_env_int("FACE_TRAINING_LIVEPORTRAIT_TIMEOUT_SECONDS", 900),
        driving_option=os.getenv(
            "FACE_TRAINING_LIVEPORTRAIT_DRIVING_OPTION",
            "expression-friendly",
        ),
        driving_multiplier=_env_float(
            "FACE_TRAINING_LIVEPORTRAIT_DRIVING_MULTIPLIER",
            1.0,
        ),
        source_max_dim=_env_int("FACE_TRAINING_LIVEPORTRAIT_SOURCE_MAX_DIM", 1280),
        source_division=_env_int("FACE_TRAINING_LIVEPORTRAIT_SOURCE_DIVISION", 2),
    )


def _frame_quality_config() -> FrameQualityConfig:
    return FrameQualityConfig(
        min_sharpness=_env_float("FACE_TRAINING_MIN_SHARPNESS", 40.0),
        min_brightness=_env_float("FACE_TRAINING_MIN_BRIGHTNESS", 40.0),
        max_brightness=_env_float("FACE_TRAINING_MAX_BRIGHTNESS", 215.0),
        min_contrast=_env_float("FACE_TRAINING_MIN_CONTRAST", 18.0),
        min_face_coverage=_env_float("FACE_TRAINING_MIN_FACE_COVERAGE", 0.05),
        max_face_coverage=_env_float("FACE_TRAINING_MAX_FACE_COVERAGE", 0.70),
        max_center_offset=_env_float("FACE_TRAINING_MAX_CENTER_OFFSET", 0.55),
    )


def _safe_filename(object_key: str) -> str:
    filename = Path(object_key).name.strip()
    return filename or "face-video.mp4"


def _boto3_client(service_name: str) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise FaceTrainingWorkerError(
            "boto3 is not installed. Install requirements.txt before running worker."
        ) from exc

    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    return boto3.client(service_name, region_name=region_name or "ap-northeast-2")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise FaceTrainingWorkerError(f"{name} must be a boolean value.")


if __name__ == "__main__":
    main()
