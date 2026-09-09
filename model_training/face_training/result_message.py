from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal


FaceTrainingStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


@dataclass(frozen=True)
class FaceTrainingResultMessage:
    job_id: int
    user_uuid: str
    clone_id: int
    status: FaceTrainingStatus
    attempt_number: int
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    occurred_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schemaVersion": 1,
                "eventType": "FACE_PROFILE_BUILD_STATUS",
                "jobId": payload.pop("job_id"),
                "userUuid": payload.pop("user_uuid"),
                "cloneId": payload.pop("clone_id"),
                "attemptNumber": payload.pop("attempt_number"),
                "occurredAt": payload.pop("occurred_at")
                or datetime.now(timezone.utc).isoformat(),
            }
        )
        if payload["result"] is None:
            payload.pop("result")
        if payload["error"] is None:
            payload.pop("error")
        return payload


def publish_face_training_result(
    sqs_client: Any,
    *,
    queue_url: str,
    message: FaceTrainingResultMessage,
) -> str:
    payload = message.to_dict()
    kwargs: dict[str, Any] = {
        "QueueUrl": queue_url,
        "MessageBody": json.dumps(payload, ensure_ascii=False),
    }
    if queue_url.lower().endswith(".fifo"):
        kwargs.update(
            {
                "MessageGroupId": message.user_uuid,
                "MessageDeduplicationId": (
                    f"face-{message.job_id}-{message.status.lower()}-"
                    f"{message.attempt_number}"
                ),
            }
        )
    response = sqs_client.send_message(**kwargs)
    return str(response.get("MessageId") or "")


def failure_detail(exc: Exception) -> dict[str, Any]:
    message = " ".join(str(exc).split())[:1000]
    return {
        "code": type(exc).__name__,
        "message": message or "Face profile build failed.",
        "retryable": True,
    }
