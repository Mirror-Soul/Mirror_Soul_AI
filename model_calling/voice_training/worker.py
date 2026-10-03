import argparse
import asyncio
import json
import mimetypes
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv

from model_calling.clone_similarity.speaker_embedding import (
    SpeakerSimilarityAudio,
    SpeakerSimilarityUnavailable,
    evaluate_speaker_similarity,
    generate_clone_reference_audio_file,
)
from model_calling.voice_training.audio_quality import (
    VoiceTrainingAudioBatch,
    VoiceTrainingAudioSample,
    VoiceTrainingInputQualityError,
    normalize_and_validate_voice_samples,
)
from model_calling.services import clone_user_voice_from_files
from model_calling.voice_training.result_message import (
    VoiceTrainingResultMessage,
    VoiceTrainingStatus,
    failure_detail,
    publish_voice_training_result,
)

load_dotenv()


class VoiceTrainingWorkerError(Exception):
    pass


@dataclass(frozen=True)
class VoiceTrainingMessage:
    job_type: str
    source: str
    job_id: int
    user_uuid: str
    bucket: str
    audio_object_keys: list[str]
    requested_at: str | None = None


@dataclass(frozen=True)
class DownloadedAudio:
    filename: str
    content: bytes
    content_type: str


_completed_results: OrderedDict[int, tuple[str, dict[str, Any]]] = OrderedDict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume Mirror Soul voice training jobs from SQS."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one available SQS message, then exit.",
    )
    args = parser.parse_args()
    run_worker(once=args.once)


def run_worker(*, once: bool = False) -> None:
    queue_url = os.getenv("AWS_SQS_VOICE_TRAINING_QUEUE_URL")
    if not queue_url:
        raise VoiceTrainingWorkerError(
            "AWS_SQS_VOICE_TRAINING_QUEUE_URL is not configured."
        )
    result_queue_url = os.getenv("AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL")
    if not result_queue_url:
        raise VoiceTrainingWorkerError(
            "AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL is not configured."
        )

    sqs_client = _boto3_client("sqs")
    s3_client = _boto3_client("s3")
    wait_seconds = _env_int("VOICE_TRAINING_WAIT_SECONDS", 20)
    visibility_timeout = _env_int("VOICE_TRAINING_VISIBILITY_TIMEOUT", 600)

    print("[VOICE_TRAINING] worker started", flush=True)
    while True:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_seconds,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages = response.get("Messages", [])

        if not messages:
            if once:
                print("[VOICE_TRAINING] no message available", flush=True)
                return
            continue

        for sqs_message in messages:
            should_delete = _handle_sqs_message(
                s3_client,
                sqs_client,
                sqs_message,
                result_queue_url=result_queue_url,
            )
            if should_delete:
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=sqs_message["ReceiptHandle"],
                )

            if once:
                return

        time.sleep(_env_float("VOICE_TRAINING_POLL_INTERVAL_SECONDS", 0.0))


def _handle_sqs_message(
    s3_client: Any,
    sqs_client: Any,
    sqs_message: dict[str, Any],
    *,
    result_queue_url: str,
) -> bool:
    try:
        message = _parse_message(sqs_message.get("Body", ""))
    except Exception as exc:
        print(f"[VOICE_TRAINING] invalid message retained: {exc}", flush=True)
        return False

    attempt_number = int(
        (sqs_message.get("Attributes") or {}).get("ApproximateReceiveCount", 1)
    )
    try:
        _publish_status(
            sqs_client,
            result_queue_url=result_queue_url,
            message=message,
            status="PROCESSING",
            attempt_number=attempt_number,
        )
    except Exception as exc:
        print(
            "[VOICE_TRAINING] processing status publish failed; request retained: "
            f"job_id={message.job_id} error={exc}",
            flush=True,
        )
        return False

    try:
        result = _cached_result(message)
        if result is None:
            result = _process_voice_training_message(s3_client, message)
            _cache_result(message, result)
        else:
            print(
                "[VOICE_TRAINING] reusing completed result: "
                f"job_id={message.job_id}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[VOICE_TRAINING] job failed: job_id={message.job_id} error={exc}",
            flush=True,
        )
        max_attempts = max(_env_int("VOICE_TRAINING_MAX_ATTEMPTS", 3), 1)
        retryable = _is_retryable_failure(exc) and attempt_number < max_attempts
        try:
            _publish_status(
                sqs_client,
                result_queue_url=result_queue_url,
                message=message,
                status="FAILED",
                attempt_number=attempt_number,
                error=failure_detail(exc, retryable=retryable),
            )
        except Exception as publish_exc:
            print(
                "[VOICE_TRAINING] failure status publish failed; request retained: "
                f"job_id={message.job_id} error={publish_exc}",
                flush=True,
            )
            return False
        return not retryable

    try:
        _publish_status(
            sqs_client,
            result_queue_url=result_queue_url,
            message=message,
            status="COMPLETED",
            attempt_number=attempt_number,
            result=result,
        )
    except Exception as exc:
        print(
            "[VOICE_TRAINING] completion status publish failed; request retained: "
            f"job_id={message.job_id} error={exc}",
            flush=True,
        )
        return False
    return True


def _publish_status(
    sqs_client: Any,
    *,
    result_queue_url: str,
    message: VoiceTrainingMessage,
    status: VoiceTrainingStatus,
    attempt_number: int,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    message_id = publish_voice_training_result(
        sqs_client,
        queue_url=result_queue_url,
        message=VoiceTrainingResultMessage(
            job_id=message.job_id,
            user_uuid=message.user_uuid,
            status=status,
            attempt_number=attempt_number,
            result=result,
            error=error,
        ),
    )
    print(
        "[VOICE_TRAINING] status published: "
        f"job_id={message.job_id} status={status} message_id={message_id}",
        flush=True,
    )


def _process_voice_training_message(
    s3_client: Any,
    message: VoiceTrainingMessage,
) -> dict[str, Any]:
    print(
        "[VOICE_TRAINING] processing: "
        f"job_id={message.job_id} user_uuid={message.user_uuid} "
        f"files={len(message.audio_object_keys)}",
        flush=True,
    )

    downloaded_files = [
        _download_audio(
            s3_client,
            bucket=message.bucket,
            object_key=object_key,
        )
        for object_key in message.audio_object_keys
    ]
    training_files = _prepare_voice_training_audio(
        downloaded_files,
        job_id=message.job_id,
    )
    voice_id = asyncio.run(
        clone_user_voice_from_files(
            message.user_uuid,
            [
                (audio.filename, audio.content, audio.content_type)
                for audio in training_files
            ],
            description=(
                "Mirror Soul voice clone "
                f"source={message.source} job_id={message.job_id}"
            ),
        )
    )
    actual_voice_score = _evaluate_actual_voice_similarity(
        original_audios=training_files,
        elevenlabs_voice_id=voice_id,
        user_uuid=message.user_uuid,
        job_id=message.job_id,
    )
    voice_score = (
        round(max(0.0, min(actual_voice_score, 100.0)), 2)
        if actual_voice_score is not None
        else _fallback_voice_score(len(training_files))
    )
    print(
        "[VOICE_TRAINING] completed: "
        f"job_id={message.job_id} "
        f"voice_id={_mask_voice_id(voice_id)} voice_score={voice_score}",
        flush=True,
    )
    return {
        "elevenlabsVoiceId": voice_id,
        "voiceScore": voice_score,
    }


def _fallback_voice_score(sample_count: int) -> float:
    expected = max(_env_int("CLONE_SIMILARITY_EXPECTED_VOICE_SAMPLES", 5), 1)
    excellent = max(
        _env_int("CLONE_SIMILARITY_EXCELLENT_VOICE_SAMPLES", 20),
        expected + 1,
    )
    if sample_count <= expected:
        return round(30.0 + 32.0 * sample_count / expected, 2)
    extra_ratio = (sample_count - expected) / (excellent - expected)
    return round(62.0 + 33.0 * min(extra_ratio, 1.0), 2)


def _cached_result(message: VoiceTrainingMessage) -> dict[str, Any] | None:
    cached = _completed_results.get(message.job_id)
    if cached is None:
        return None
    cached_user_uuid, result = cached
    if cached_user_uuid != message.user_uuid:
        raise VoiceTrainingWorkerError(
            f"jobId {message.job_id} was already used by another userUuid"
        )
    _completed_results.move_to_end(message.job_id)
    return dict(result)


def _cache_result(
    message: VoiceTrainingMessage,
    result: dict[str, Any],
) -> None:
    _completed_results[message.job_id] = (message.user_uuid, dict(result))
    _completed_results.move_to_end(message.job_id)
    cache_size = max(_env_int("VOICE_TRAINING_IDEMPOTENCY_CACHE_SIZE", 1000), 1)
    while len(_completed_results) > cache_size:
        _completed_results.popitem(last=False)


def _is_retryable_failure(exc: Exception) -> bool:
    if isinstance(exc, VoiceTrainingInputQualityError):
        return False
    message = str(exc).lower()
    non_retryable_markers = (
        "voice_input_quality_failed",
        "empty s3 object",
        "audioobjectkeys",
        "invalid audioobjectkey",
        "must be",
        "unsupported",
    )
    return not any(marker in message for marker in non_retryable_markers)


def _prepare_voice_training_audio(
    downloaded_files: list[DownloadedAudio],
    *,
    job_id: int,
) -> list[DownloadedAudio]:
    if not _env_bool("VOICE_TRAINING_AUDIO_QUALITY_ENABLED", True):
        print(
            "[VOICE_TRAINING_QUALITY] bypassed: "
            f"job_id={job_id} files={len(downloaded_files)}",
            flush=True,
        )
        return downloaded_files

    try:
        batch = normalize_and_validate_voice_samples(
            [
                VoiceTrainingAudioSample(
                    filename=audio.filename,
                    content=audio.content,
                    content_type=audio.content_type,
                )
                for audio in downloaded_files
            ]
        )
    except VoiceTrainingInputQualityError as exc:
        _log_voice_training_quality(exc.batch, job_id=job_id, passed=False)
        raise VoiceTrainingWorkerError(str(exc)) from exc

    _log_voice_training_quality(batch, job_id=job_id, passed=True)
    return [
        DownloadedAudio(
            filename=sample.filename,
            content=sample.content,
            content_type=sample.content_type,
        )
        for sample in batch.accepted
    ]


def _log_voice_training_quality(
    batch: VoiceTrainingAudioBatch,
    *,
    job_id: int,
    passed: bool,
) -> None:
    for sample in batch.accepted:
        metrics = sample.metrics
        print(
            "[VOICE_TRAINING_QUALITY] sample: "
            f"job_id={job_id} sample={sample.sample_number} status=ACCEPTED "
            f"duration={metrics.duration_seconds:.2f}s "
            f"rms_dbfs={metrics.rms_dbfs:.2f} "
            f"silence_ratio={metrics.silence_ratio:.3f} "
            f"clipping_ratio={metrics.clipping_ratio:.4f}",
            flush=True,
        )
    for sample in batch.rejected:
        metrics_text = ""
        if sample.metrics is not None:
            metrics_text = (
                f" duration={sample.metrics.duration_seconds:.2f}s"
                f" rms_dbfs={sample.metrics.rms_dbfs:.2f}"
                f" silence_ratio={sample.metrics.silence_ratio:.3f}"
                f" clipping_ratio={sample.metrics.clipping_ratio:.4f}"
            )
        print(
            "[VOICE_TRAINING_QUALITY] sample: "
            f"job_id={job_id} sample={sample.sample_number} status=REJECTED "
            f"reasons={','.join(sample.reason_codes)}{metrics_text}",
            flush=True,
        )
    print(
        "[VOICE_TRAINING_QUALITY] batch: "
        f"job_id={job_id} status={'PASSED' if passed else 'FAILED'} "
        f"accepted={len(batch.accepted)} rejected={len(batch.rejected)} "
        f"duration={batch.total_duration_seconds:.2f}s",
        flush=True,
    )


def _evaluate_actual_voice_similarity(
    *,
    original_audios: list[DownloadedAudio],
    elevenlabs_voice_id: str,
    user_uuid: str,
    job_id: int,
) -> float | None:
    reference_audio_path = _build_clone_reference_audio_path(
        user_uuid=user_uuid,
        job_id=job_id,
    )
    try:
        generate_clone_reference_audio_file(
            elevenlabs_voice_id=elevenlabs_voice_id,
            output_path=reference_audio_path,
        )
        print(
            "[CLONE_SIMILARITY] reference audio saved: "
            f"path={reference_audio_path}",
            flush=True,
        )
    except SpeakerSimilarityUnavailable as exc:
        print(
            "[CLONE_SIMILARITY] reference audio generation skipped: "
            f"{exc}",
            flush=True,
        )
        return None
    except Exception as exc:
        print(
            "[CLONE_SIMILARITY] reference audio generation failed: "
            f"{exc}",
            flush=True,
        )
        return None

    try:
        result = evaluate_speaker_similarity(
            original_audios=[
                SpeakerSimilarityAudio(
                    filename=audio.filename,
                    content=audio.content,
                    content_type=audio.content_type,
                )
                for audio in original_audios
            ],
            elevenlabs_voice_id=elevenlabs_voice_id,
            reference_audio_path=reference_audio_path,
        )
        print(
            "[CLONE_SIMILARITY] speaker embedding result: "
            f"score={result.score} cosine={result.cosine_similarity} "
            f"samples={result.original_sample_count} model={result.model_name} "
            f"reference_audio={result.reference_audio_path}",
            flush=True,
        )
        return result.score
    except SpeakerSimilarityUnavailable as exc:
        print(
            "[CLONE_SIMILARITY] speaker embedding skipped: "
            f"{exc}",
            flush=True,
        )
        return None
    except Exception as exc:
        print(
            "[CLONE_SIMILARITY] speaker embedding failed; "
            f"using fallback score: {exc}",
            flush=True,
        )
        return None


def _build_clone_reference_audio_path(*, user_uuid: str, job_id: int) -> Path:
    base_dir = os.getenv(
        "CLONE_SIMILARITY_REFERENCE_AUDIO_DIR",
        "model_calling/assets/clone_similarity",
    )
    return Path(base_dir) / user_uuid / f"job-{job_id}-reference.mp3"


def _parse_message(message_body: str) -> VoiceTrainingMessage:
    try:
        data = json.loads(message_body)
    except json.JSONDecodeError as exc:
        raise VoiceTrainingWorkerError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise VoiceTrainingWorkerError("Message body must be a JSON object.")

    missing_fields = [
        field
        for field in (
            "jobType",
            "source",
            "jobId",
            "userUuid",
            "bucket",
            "audioObjectKeys",
        )
        if data.get(field) in (None, "")
    ]
    if missing_fields:
        raise VoiceTrainingWorkerError(
            f"Missing required field(s): {', '.join(missing_fields)}"
        )

    try:
        job_id = int(data["jobId"])
    except (TypeError, ValueError) as exc:
        raise VoiceTrainingWorkerError("jobId must be an integer.") from exc
    if str(data["jobType"]) != "VOICE_TRAINING":
        raise VoiceTrainingWorkerError("Unsupported voice training message type.")
    if job_id <= 0:
        raise VoiceTrainingWorkerError("jobId must be positive.")
    user_uuid = str(data["userUuid"])
    try:
        UUID(user_uuid)
    except ValueError as exc:
        raise VoiceTrainingWorkerError("userUuid must be a valid UUID.") from exc
    audio_object_keys = data["audioObjectKeys"]
    if not isinstance(audio_object_keys, list) or not audio_object_keys:
        raise VoiceTrainingWorkerError("audioObjectKeys must be a non-empty list.")
    normalized_keys = []
    for object_key in audio_object_keys:
        if not isinstance(object_key, str):
            raise VoiceTrainingWorkerError("audioObjectKeys must contain strings.")
        normalized = object_key.strip().replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise VoiceTrainingWorkerError(f"Invalid audioObjectKey: {object_key}")
        normalized_keys.append(normalized)

    return VoiceTrainingMessage(
        job_type="VOICE_TRAINING",
        source=str(data["source"]),
        job_id=job_id,
        user_uuid=user_uuid,
        bucket=str(data["bucket"]),
        audio_object_keys=normalized_keys,
        requested_at=data.get("requestedAt"),
    )


def _download_audio(
    s3_client: Any,
    *,
    bucket: str,
    object_key: str,
) -> DownloadedAudio:
    response = s3_client.get_object(Bucket=bucket, Key=object_key)
    content = response["Body"].read()
    if not content:
        raise VoiceTrainingWorkerError(f"Empty S3 object: s3://{bucket}/{object_key}")

    filename = object_key.rsplit("/", 1)[-1] or "voice-sample.wav"
    content_type = response.get("ContentType") or _guess_content_type(filename)
    return DownloadedAudio(
        filename=filename,
        content=content,
        content_type=content_type,
    )


def _guess_content_type(filename: str) -> str:
    guessed_content_type, _ = mimetypes.guess_type(filename)
    return guessed_content_type or "audio/wav"


def _boto3_client(service_name: str) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise VoiceTrainingWorkerError(
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
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _mask_voice_id(voice_id: str) -> str:
    if len(voice_id) <= 8:
        return "set"
    return f"{voice_id[:4]}...{voice_id[-4:]}"


if __name__ == "__main__":
    main()
