from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel, StrictInt, StrictStr

from shared.config import settings


_RETRYABLE_STATUS_CODES = frozenset({502, 504})


class PersonaContext(BaseModel):
    name: str | None = None
    gender: str | None = None
    birthDate: date | None = None
    job: str | None = None
    jobDescription: str | None = None
    selfIntroduction: str | None = None
    mbti: str | None = None


class VoiceContext(BaseModel):
    voiceProfileId: StrictInt
    voiceTrainingJobId: StrictInt
    provider: StrictStr
    voiceId: StrictStr


class CloneContext(BaseModel):
    cloneId: StrictInt
    userUuid: StrictStr
    persona: PersonaContext
    voice: VoiceContext


class CallContext(BaseModel):
    schemaVersion: StrictInt
    callId: StrictInt
    roomId: StrictStr
    mediaType: StrictStr
    status: StrictStr
    clone: CloneContext


class CallContextError(RuntimeError):
    reject_reason = "INVALID_CALL_CONTEXT"

    def __init__(self, message: str, *, backend_code: str | None = None) -> None:
        super().__init__(message)
        self.backend_code = backend_code


class CallContextNotFound(CallContextError):
    reject_reason = "CALL_CONTEXT_NOT_FOUND"


class InvalidCallStatus(CallContextError):
    reject_reason = "INVALID_CALL_STATUS"


class CloneNotReady(CallContextError):
    reject_reason = "CLONE_NOT_READY"


class CallContextConfigurationError(CallContextError):
    reject_reason = "AI_SERVER_CONFIG_ERROR"


class CallContextUnavailable(CallContextError):
    reject_reason = "CALL_CONTEXT_UNAVAILABLE"


class InvalidCallContext(CallContextError):
    reject_reason = "INVALID_CALL_CONTEXT"


def _parse_model(model_type: type[BaseModel], value: Any) -> BaseModel:
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)
    return model_type.parse_obj(value)


def _backend_error(response: httpx.Response) -> tuple[str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    code = body.get("code")
    message = body.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


def _raise_for_backend_error(response: httpx.Response) -> None:
    code, backend_message = _backend_error(response)
    message = backend_message or f"Backend returned HTTP {response.status_code}."
    if response.status_code == 404 or code == "CALL_4040":
        raise CallContextNotFound(message, backend_code=code)
    if code == "CALL_4090":
        raise InvalidCallStatus(message, backend_code=code)
    if code in {"CLONE_4090", "VOICE_4090"}:
        raise CloneNotReady(message, backend_code=code)
    if response.status_code in {401, 503} or code in {
        "INTERNAL_4010",
        "INTERNAL_5030",
    }:
        raise CallContextConfigurationError(message, backend_code=code)
    if response.status_code >= 500:
        raise CallContextUnavailable(message, backend_code=code)
    raise InvalidCallContext(message, backend_code=code)


def _validate_context(context: CallContext) -> CallContext:
    errors: list[str] = []
    if context.schemaVersion != 1:
        errors.append("schemaVersion must be 1")
    if context.callId <= 0:
        errors.append("callId must be positive")
    if not context.roomId.strip():
        errors.append("roomId must not be empty")
    if context.mediaType not in {"VOICE", "VIDEO"}:
        errors.append("mediaType is not supported")
    if context.status not in {"READY", "IN_PROGRESS"}:
        errors.append("status is not allowed")
    if context.clone.cloneId <= 0:
        errors.append("cloneId must be positive")
    try:
        UUID(context.clone.userUuid)
    except (TypeError, ValueError, AttributeError):
        errors.append("userUuid must be a UUID")
    voice = context.clone.voice
    if voice.voiceProfileId <= 0:
        errors.append("voiceProfileId must be positive")
    if voice.voiceTrainingJobId <= 0:
        errors.append("voiceTrainingJobId must be positive")
    if not voice.provider.strip():
        errors.append("provider must not be empty")
    if not voice.voiceId.strip():
        errors.append("voiceId must not be empty")
    if errors:
        raise InvalidCallContext("; ".join(errors))
    return context


async def fetch_call_context(
    call_id: int,
    *,
    client: httpx.AsyncClient | None = None,
) -> CallContext:
    base_url = settings.BACKEND_API_BASE_URL
    api_key = settings.AI_INTERNAL_API_KEY
    timeout_seconds = settings.BACKEND_CALL_CONTEXT_TIMEOUT_SECONDS
    if not base_url:
        raise CallContextConfigurationError("BACKEND_API_BASE_URL is not configured.")
    if not api_key:
        raise CallContextConfigurationError("AI_INTERNAL_API_KEY is not configured.")
    if timeout_seconds <= 0:
        raise CallContextConfigurationError(
            "BACKEND_CALL_CONTEXT_TIMEOUT_SECONDS must be positive."
        )

    url = f"{base_url.rstrip('/')}/internal/ai/calls/{call_id}/context"
    max_attempts = max(1, int(settings.BACKEND_CALL_CONTEXT_MAX_ATTEMPTS))
    backoff_seconds = max(0.0, float(settings.BACKEND_CALL_CONTEXT_RETRY_BACKOFF_SECONDS))
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        # GET is idempotent, so only transient transport failures and gateway
        # errors are retried a limited number of times. Business errors (404,
        # 409, 401, ...) are never retried.
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await http_client.get(
                    url,
                    headers={"X-Internal-Api-Key": api_key},
                    timeout=timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_seconds * attempt)
                    continue
                raise CallContextUnavailable(
                    "Backend call context is unavailable."
                ) from exc
            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < max_attempts
            ):
                await asyncio.sleep(backoff_seconds * attempt)
                continue
            break
    finally:
        if owns_client:
            await http_client.aclose()

    if not response.is_success:
        _raise_for_backend_error(response)

    try:
        body = response.json()
    except ValueError as exc:
        raise InvalidCallContext("Backend returned invalid JSON.") from exc
    if not isinstance(body, dict):
        raise InvalidCallContext("Backend response must be a JSON object.")
    if body.get("isSuccess") is not True:
        code = body.get("code") if isinstance(body.get("code"), str) else None
        raise InvalidCallContext(
            "Backend response did not indicate success.",
            backend_code=code,
        )
    result = body.get("result")
    if result is None:
        raise InvalidCallContext("Backend response result is missing.")
    try:
        context = _parse_model(CallContext, result)
    except Exception as exc:
        raise InvalidCallContext("Backend call context schema is invalid.") from exc
    return _validate_context(context)  # type: ignore[arg-type]
