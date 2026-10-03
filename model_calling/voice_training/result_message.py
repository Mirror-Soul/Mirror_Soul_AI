from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


VoiceTrainingStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


@dataclass(frozen=True)
class VoiceTrainingResultMessage:
    job_id: int
    user_uuid: str
    status: VoiceTrainingStatus
    attempt_number: int
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = _validated_result(self.status, self.result)
        error = _validated_error(self.status, self.error)
        payload: dict[str, Any] = {
            "eventType": "VOICE_TRAINING_STATUS",
            "jobId": self.job_id,
            "userUuid": self.user_uuid,
            "status": self.status,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        return payload


def publish_voice_training_result(
    sqs_client: Any,
    *,
    queue_url: str,
    message: VoiceTrainingResultMessage,
) -> str:
    kwargs: dict[str, Any] = {
        "QueueUrl": queue_url,
        "MessageBody": json.dumps(message.to_dict(), ensure_ascii=False),
    }
    if queue_url.lower().endswith(".fifo"):
        kwargs.update(
            {
                "MessageGroupId": message.user_uuid,
                "MessageDeduplicationId": (
                    f"voice-{message.job_id}-{message.status.lower()}-"
                    f"{message.attempt_number}"
                ),
            }
        )
    response = sqs_client.send_message(**kwargs)
    return str(response.get("MessageId") or "")


def failure_detail(exc: Exception, *, retryable: bool) -> dict[str, Any]:
    message = " ".join(str(exc).split())[:1000]
    return {
        "code": type(exc).__name__,
        "message": message or "Voice profile build failed.",
        "retryable": retryable,
    }


def _validated_result(
    status: VoiceTrainingStatus,
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if status != "COMPLETED":
        return None
    if not isinstance(result, dict):
        raise ValueError("COMPLETED voice result is required")
    voice_id = str(result.get("elevenlabsVoiceId") or "").strip()
    score = result.get("voiceScore")
    if not voice_id:
        raise ValueError("elevenlabsVoiceId is required")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("voiceScore must be a number")
    normalized_score = round(float(score), 2)
    if not 0.0 <= normalized_score <= 100.0:
        raise ValueError("voiceScore must be between 0 and 100")

    normalized = dict(result)
    normalized["elevenlabsVoiceId"] = voice_id
    normalized["voiceScore"] = normalized_score
    intro_audio = normalized.get("introAudio")
    if intro_audio is not None:
        if not isinstance(intro_audio, dict):
            raise ValueError("introAudio must be an object")
        required = ("bucket", "objectKey", "contentType", "sizeBytes", "durationMs")
        missing = [name for name in required if intro_audio.get(name) in (None, "")]
        if missing:
            raise ValueError(
                f"introAudio missing required field(s): {', '.join(missing)}"
            )
    return normalized


def _validated_error(
    status: VoiceTrainingStatus,
    error: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if status != "FAILED":
        return None
    if not isinstance(error, dict):
        raise ValueError("FAILED voice error is required")
    if not str(error.get("code") or "").strip():
        raise ValueError("error.code is required")
    if not str(error.get("message") or "").strip():
        raise ValueError("error.message is required")
    if not isinstance(error.get("retryable"), bool):
        raise ValueError("error.retryable must be a boolean")
    return dict(error)
