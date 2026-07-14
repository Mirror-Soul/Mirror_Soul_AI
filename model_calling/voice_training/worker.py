import argparse
import asyncio
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from model_calling.repository.clone_repository import (
    CloneRepositoryError,
    complete_voice_training_job,
    find_clone_by_user_uuid,
    find_voice_training_job_files,
    find_voice_training_job_status,
    mark_voice_training_job_failed,
    mark_voice_training_job_processing,
)
from model_calling.services import clone_user_voice_from_files

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
        )
        messages = response.get("Messages", [])

        if not messages:
            if once:
                print("[VOICE_TRAINING] no message available", flush=True)
                return
            continue

        for sqs_message in messages:
            should_delete = _handle_sqs_message(s3_client, sqs_message)
            if should_delete:
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=sqs_message["ReceiptHandle"],
                )

            if once:
                return

        time.sleep(_env_float("VOICE_TRAINING_POLL_INTERVAL_SECONDS", 0.0))


def _handle_sqs_message(s3_client: Any, sqs_message: dict[str, Any]) -> bool:
    try:
        message = _parse_message(sqs_message.get("Body", ""))
    except Exception as exc:
        print(f"[VOICE_TRAINING] invalid message skipped: {exc}", flush=True)
        return True

    try:
        _process_voice_training_message(s3_client, message)
        return True
    except Exception as exc:
        print(
            f"[VOICE_TRAINING] job failed: job_id={message.job_id} error={exc}",
            flush=True,
        )
        try:
            mark_voice_training_job_failed(message.job_id, str(exc))
        except CloneRepositoryError as db_exc:
            print(
                "[VOICE_TRAINING] failed to update job failure status: "
                f"job_id={message.job_id} error={db_exc}",
                flush=True,
            )
        return _env_bool("VOICE_TRAINING_DELETE_FAILED_MESSAGES", True)


def _process_voice_training_message(
    s3_client: Any,
    message: VoiceTrainingMessage,
) -> None:
    if message.job_type != "VOICE_TRAINING":
        raise VoiceTrainingWorkerError(f"Unsupported jobType: {message.job_type}")

    current_status = find_voice_training_job_status(message.job_id)
    if current_status == "COMPLETED":
        print(
            f"[VOICE_TRAINING] job already completed: job_id={message.job_id}",
            flush=True,
        )
        return
    if current_status is None:
        raise VoiceTrainingWorkerError(f"voice_training_job not found: {message.job_id}")

    clone = find_clone_by_user_uuid(message.user_uuid)
    audio_sources = _resolve_audio_sources(message)
    if not audio_sources:
        raise VoiceTrainingWorkerError(
            f"No audio files found for job_id={message.job_id}"
        )

    mark_voice_training_job_processing(message.job_id)
    print(
        "[VOICE_TRAINING] processing: "
        f"job_id={message.job_id} user_uuid={message.user_uuid} "
        f"clone_id={clone.clone_id} files={len(audio_sources)}",
        flush=True,
    )

    downloaded_files = [
        _download_audio(s3_client, bucket=bucket, object_key=object_key)
        for bucket, object_key in audio_sources
    ]
    voice_id = asyncio.run(
        clone_user_voice_from_files(
            message.user_uuid,
            [
                (audio.filename, audio.content, audio.content_type)
                for audio in downloaded_files
            ],
            description=(
                "Mirror Soul voice clone "
                f"source={message.source} job_id={message.job_id}"
            ),
        )
    )

    complete_voice_training_job(
        job_id=message.job_id,
        clone_id=clone.clone_id,
        elevenlabs_voice_id=voice_id,
    )
    print(
        "[VOICE_TRAINING] completed: "
        f"job_id={message.job_id} clone_id={clone.clone_id} "
        f"voice_id={_mask_voice_id(voice_id)}",
        flush=True,
    )


def _parse_message(message_body: str) -> VoiceTrainingMessage:
    data = json.loads(message_body)
    audio_object_keys = data.get("audioObjectKeys") or []
    if not isinstance(audio_object_keys, list):
        raise VoiceTrainingWorkerError("audioObjectKeys must be a list.")

    missing_fields = [
        field
        for field in ("jobType", "source", "jobId", "userUuid", "bucket")
        if data.get(field) in (None, "")
    ]
    if missing_fields:
        raise VoiceTrainingWorkerError(
            f"Missing required field(s): {', '.join(missing_fields)}"
        )

    return VoiceTrainingMessage(
        job_type=str(data["jobType"]),
        source=str(data["source"]),
        job_id=int(data["jobId"]),
        user_uuid=str(data["userUuid"]),
        bucket=str(data["bucket"]),
        audio_object_keys=[str(object_key) for object_key in audio_object_keys],
        requested_at=data.get("requestedAt"),
    )


def _resolve_audio_sources(message: VoiceTrainingMessage) -> list[tuple[str, str]]:
    if message.audio_object_keys:
        return [(message.bucket, object_key) for object_key in message.audio_object_keys]

    return [
        (job_file.bucket, job_file.object_key)
        for job_file in find_voice_training_job_files(message.job_id)
    ]


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
