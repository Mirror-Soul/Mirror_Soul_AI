"""Client for the backend call history (talk log) API.

The call server sends every finished utterance of a realtime call to
``POST /internal/ai/calls/{callId}/talk-logs`` so the app can show the
conversation later through ``GET /history/calls/{callId}/talk-logs``.

The backend deduplicates by ``eventId``, so a request that timed out can be
sent again with the same entry without creating a second row.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx

from shared.config import settings


TalkLogSpeaker = Literal["USER", "CLONE"]

# Backend validation: @Size(max = 2000) on message.
MAX_MESSAGE_CHARS = 2000

# Business errors (4xx) are final. Gateway and server errors are retried
# because the request is idempotent by eventId.
_NON_RETRYABLE_BACKEND_CODES = frozenset({"INTERNAL_5030"})


@dataclass(frozen=True)
class TalkLogEntry:
    speaker: TalkLogSpeaker
    message: str
    started_at: datetime
    ended_at: datetime | None = None
    turn_id: int | None = None
    event_id: UUID = field(default_factory=uuid4)

    def payload(self) -> dict[str, Any]:
        started_at = _as_utc(self.started_at)
        ended_at = _as_utc(self.ended_at) if self.ended_at is not None else None
        if ended_at is not None and ended_at < started_at:
            ended_at = started_at
        return {
            "eventId": str(self.event_id),
            "speaker": self.speaker,
            "message": normalize_message(self.message),
            "startedAt": _iso(started_at),
            "endedAt": _iso(ended_at) if ended_at is not None else None,
        }


@dataclass(frozen=True)
class TalkLogSaveResult:
    talk_log_id: int | None
    duplicated: bool


class TalkLogSaveError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def normalize_message(message: str) -> str:
    return message.strip()[:MAX_MESSAGE_CHARS]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def talk_log_config_problem() -> str | None:
    """Return why talk logs cannot be sent, or ``None`` when configured."""
    if not settings.BACKEND_TALK_LOG_ENABLED:
        return "BACKEND_TALK_LOG_ENABLED=false"
    if not settings.BACKEND_API_BASE_URL:
        return "BACKEND_API_BASE_URL is not configured"
    if not settings.AI_INTERNAL_API_KEY:
        return "AI_INTERNAL_API_KEY is not configured"
    if settings.BACKEND_TALK_LOG_TIMEOUT_SECONDS <= 0:
        return "BACKEND_TALK_LOG_TIMEOUT_SECONDS must be positive"
    return None


def _backend_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("code"), str):
        return body["code"]
    return None


def _error_from_response(response: httpx.Response) -> TalkLogSaveError:
    code = _backend_code(response) or f"HTTP_{response.status_code}"
    retryable = (
        response.status_code >= 500 and code not in _NON_RETRYABLE_BACKEND_CODES
    )
    return TalkLogSaveError(
        f"Backend rejected talk log with HTTP {response.status_code}.",
        code=code,
        retryable=retryable,
    )


def _parse_result(response: httpx.Response) -> TalkLogSaveResult:
    try:
        body = response.json()
    except ValueError as exc:
        raise TalkLogSaveError(
            "Backend returned invalid JSON.",
            code="INVALID_RESPONSE",
        ) from exc
    if not isinstance(body, dict) or body.get("isSuccess") is not True:
        code = body.get("code") if isinstance(body, dict) else None
        raise TalkLogSaveError(
            "Backend response did not indicate success.",
            code=code if isinstance(code, str) else "INVALID_RESPONSE",
        )
    result = body.get("result")
    if not isinstance(result, dict):
        return TalkLogSaveResult(talk_log_id=None, duplicated=False)
    talk_log_id = result.get("talkLogId")
    return TalkLogSaveResult(
        talk_log_id=talk_log_id if isinstance(talk_log_id, int) else None,
        duplicated=result.get("duplicated") is True,
    )


async def save_talk_log(
    call_id: int,
    entry: TalkLogEntry,
    *,
    client: httpx.AsyncClient | None = None,
) -> TalkLogSaveResult:
    problem = talk_log_config_problem()
    if problem is not None:
        raise TalkLogSaveError(problem, code="AI_SERVER_CONFIG_ERROR")
    if not normalize_message(entry.message):
        raise TalkLogSaveError("Talk log message is empty.", code="EMPTY_MESSAGE")

    base_url = settings.BACKEND_API_BASE_URL
    url = f"{base_url.rstrip('/')}/internal/ai/calls/{call_id}/talk-logs"
    timeout_seconds = settings.BACKEND_TALK_LOG_TIMEOUT_SECONDS
    max_attempts = max(1, int(settings.BACKEND_TALK_LOG_MAX_ATTEMPTS))
    backoff_seconds = max(0.0, float(settings.BACKEND_TALK_LOG_RETRY_BACKOFF_SECONDS))
    headers = {"X-Internal-Api-Key": settings.AI_INTERNAL_API_KEY}
    payload = entry.payload()

    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await http_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_seconds * attempt)
                    continue
                raise TalkLogSaveError(
                    f"Backend talk log API is unavailable: {type(exc).__name__}",
                    code="BACKEND_UNAVAILABLE",
                    retryable=True,
                ) from exc

            if response.is_success:
                return _parse_result(response)
            error = _error_from_response(response)
            if error.retryable and attempt < max_attempts:
                await asyncio.sleep(backoff_seconds * attempt)
                continue
            raise error
    finally:
        if owns_client:
            await http_client.aclose()
