from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from model_calling.realtime.video import QueuedVideoTrack


PROFILE_MAX_BYTES = 128 * 1024
PORTRAIT_MAX_BYTES = 10 * 1024 * 1024


class DittoRealtimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DittoCallConfig:
    service_url: str
    api_key: str
    timeout_seconds: float = 300.0
    max_response_bytes: int = 100 * 1024 * 1024
    retry_attempts: int = 3
    retry_base_seconds: float = 0.5

    @classmethod
    def from_env(cls) -> "DittoCallConfig | None":
        service_url = os.getenv("DITTO_CALL_SERVICE_URL", "").strip()
        api_key = os.getenv("DITTO_CALL_SERVICE_API_KEY", "").strip()
        if not service_url and not api_key:
            return None
        if not service_url or not api_key:
            raise DittoRealtimeError(
                "DITTO_CALL_SERVICE_URL and DITTO_CALL_SERVICE_API_KEY "
                "must be configured together."
            )

        parsed = urlparse(service_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise DittoRealtimeError("DITTO_CALL_SERVICE_URL is invalid.")
        is_loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            parsed.scheme == "http"
            and not is_loopback
            and not _env_bool("DITTO_CALL_ALLOW_INSECURE_HTTP", False)
        ):
            raise DittoRealtimeError(
                "Non-loopback Ditto connections require HTTPS or an explicit "
                "DITTO_CALL_ALLOW_INSECURE_HTTP=true override for an encrypted "
                "private tunnel."
            )

        timeout_seconds = _env_float("DITTO_CALL_TIMEOUT_SECONDS", 300.0)
        max_response_bytes = _env_int(
            "DITTO_CALL_MAX_RESPONSE_BYTES",
            100 * 1024 * 1024,
        )
        retry_attempts = _env_int("DITTO_CALL_RETRY_ATTEMPTS", 3)
        retry_base_seconds = _env_float(
            "DITTO_CALL_RETRY_BASE_SECONDS",
            0.5,
        )
        if (
            timeout_seconds <= 0
            or max_response_bytes <= 0
            or retry_attempts <= 0
            or retry_base_seconds < 0
        ):
            raise DittoRealtimeError(
                "Ditto call limits and retry settings are invalid."
            )
        return cls(
            service_url=service_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            retry_attempts=retry_attempts,
            retry_base_seconds=retry_base_seconds,
        )


@dataclass(frozen=True)
class FaceRenderProfile:
    portrait_bytes: bytes
    portrait_filename: str
    portrait_content_type: str
    profile_bytes: bytes | None = None


class DittoRenderClient:
    def __init__(
        self,
        config: DittoCallConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    async def render(
        self,
        profile: FaceRenderProfile,
        audio_bytes: bytes,
    ) -> bytes:
        if not audio_bytes:
            raise DittoRealtimeError("Ditto reply audio must not be empty.")
        files = {
            "portrait": (
                profile.portrait_filename,
                profile.portrait_bytes,
                profile.portrait_content_type,
            ),
            "audio": ("reply.mp3", audio_bytes, "audio/mpeg"),
        }
        if profile.profile_bytes is not None:
            files["profile"] = (
                "face-profile.json",
                profile.profile_bytes,
                "application/json",
            )

        async def send(client: httpx.AsyncClient) -> httpx.Response:
            response = None
            for attempt in range(1, self.config.retry_attempts + 1):
                try:
                    response = await client.post(
                        f"{self.config.service_url}/api/v1/render",
                        headers={"X-Ditto-Api-Key": self.config.api_key},
                        files=files,
                    )
                except httpx.HTTPError as exc:
                    raise DittoRealtimeError(
                        f"Ditto render request failed: {exc}"
                    ) from exc
                if response.status_code != 429:
                    return response
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_base_seconds * (2 ** (attempt - 1))
                    print(
                        "[DITTO_CALL] GPU busy; retrying render: "
                        f"attempt={attempt} delay={delay:.2f}s",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
            assert response is not None
            return response

        if self._http_client is not None:
            response = await send(self._http_client)
        else:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds
            ) as client:
                response = await send(client)

        if response.status_code != 200:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise DittoRealtimeError(
                f"Ditto render request failed: status={response.status_code} "
                f"detail={detail or 'empty response'}"
            )
        content_type = response.headers.get("content-type", "")
        if not content_type.lower().startswith("video/mp4"):
            raise DittoRealtimeError(
                f"Ditto response is not MP4: content_type={content_type or 'missing'}"
            )
        if not response.content:
            raise DittoRealtimeError("Ditto response MP4 is empty.")
        if len(response.content) > self.config.max_response_bytes:
            raise DittoRealtimeError(
                "Ditto response exceeds "
                f"{self.config.max_response_bytes} bytes."
            )
        print(
            "[DITTO_CALL] render completed: "
            f"bytes={len(response.content)} "
            f"seconds={response.headers.get('x-ditto-render-seconds', 'unknown')}",
            flush=True,
        )
        return response.content


class FaceProfileLoader:
    def __init__(self, *, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client

    async def load(self, user_id: str, clone_id: int) -> FaceRenderProfile:
        return await asyncio.to_thread(self._load, user_id, clone_id)

    def _load(self, user_id: str, clone_id: int) -> FaceRenderProfile:
        local_portrait = os.getenv("DITTO_CALL_LOCAL_PORTRAIT_PATH", "").strip()
        if local_portrait:
            return self._load_local(Path(local_portrait), user_id, clone_id)
        return self._load_s3(user_id, clone_id)

    def _load_local(
        self,
        portrait_path: Path,
        user_id: str,
        clone_id: int,
    ) -> FaceRenderProfile:
        portrait_bytes = _read_local_file(
            portrait_path,
            max_bytes=PORTRAIT_MAX_BYTES,
            label="Ditto local portrait",
        )
        profile_bytes = None
        profile_path_value = os.getenv(
            "DITTO_CALL_LOCAL_PROFILE_PATH",
            "",
        ).strip()
        if profile_path_value:
            profile_bytes = _read_local_file(
                Path(profile_path_value),
                max_bytes=PROFILE_MAX_BYTES,
                label="Ditto local face profile",
            )
            _validate_profile(profile_bytes, user_id, clone_id)

        return FaceRenderProfile(
            portrait_bytes=portrait_bytes,
            portrait_filename=portrait_path.name,
            portrait_content_type=(
                mimetypes.guess_type(portrait_path.name)[0] or "image/jpeg"
            ),
            profile_bytes=profile_bytes,
        )

    def _load_s3(self, user_id: str, clone_id: int) -> FaceRenderProfile:
        bucket = (
            os.getenv("DITTO_CALL_S3_BUCKET")
            or os.getenv("AWS_S3_BUCKET")
            or ""
        ).strip()
        if not bucket:
            raise DittoRealtimeError(
                "No Ditto portrait source is configured. Set local portrait "
                "paths or DITTO_CALL_S3_BUCKET."
            )
        result_prefix = os.getenv(
            "DITTO_CALL_FACE_RESULT_PREFIX",
            os.getenv("FACE_TRAINING_RESULT_PREFIX", "face-results"),
        ).strip().strip("/")
        if not result_prefix or ".." in result_prefix.split("/"):
            raise DittoRealtimeError("DITTO_CALL_FACE_RESULT_PREFIX is invalid.")

        s3_client = self._s3_client or _create_s3_client()
        prefix = f"{result_prefix}/{user_id}/"
        candidates: list[dict[str, Any]] = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            candidates.extend(
                item
                for item in page.get("Contents", [])
                if str(item.get("Key", "")).endswith("/face-profile.json")
            )
        if not candidates:
            raise DittoRealtimeError(
                f"No face profile found for user={user_id}."
            )
        profile_object = max(candidates, key=_s3_last_modified)
        profile_key = str(profile_object["Key"])
        profile_bytes, _ = _read_s3_object(
            s3_client,
            bucket=bucket,
            object_key=profile_key,
            max_bytes=PROFILE_MAX_BYTES,
            label="Ditto face profile",
        )
        profile_data = _validate_profile(profile_bytes, user_id, clone_id)
        portrait = profile_data.get("portrait") or {}
        portrait_bucket = str(portrait.get("bucket") or bucket)
        portrait_key = str(portrait.get("objectKey") or "")
        if not portrait_key:
            raise DittoRealtimeError(
                "Face profile does not contain portrait.objectKey."
            )
        portrait_bytes, response_content_type = _read_s3_object(
            s3_client,
            bucket=portrait_bucket,
            object_key=portrait_key,
            max_bytes=PORTRAIT_MAX_BYTES,
            label="Ditto portrait",
        )
        content_type = str(
            portrait.get("contentType")
            or response_content_type
            or mimetypes.guess_type(portrait_key)[0]
            or "image/jpeg"
        )
        print(
            "[DITTO_CALL] face profile loaded: "
            f"user={user_id} clone_id={clone_id} key={profile_key}",
            flush=True,
        )
        return FaceRenderProfile(
            portrait_bytes=portrait_bytes,
            portrait_filename=Path(portrait_key).name or "portrait.jpg",
            portrait_content_type=content_type,
            profile_bytes=profile_bytes,
        )


class DittoVideoSession:
    def __init__(
        self,
        *,
        user_id: str,
        clone_id: int,
        track: "QueuedVideoTrack",
        client: DittoRenderClient,
        profile_loader: FaceProfileLoader,
    ) -> None:
        self.user_id = user_id
        self.clone_id = clone_id
        self.track = track
        self.client = client
        self.profile_loader = profile_loader
        self._profile: FaceRenderProfile | None = None
        self._profile_lock = asyncio.Lock()

    async def prepare(self) -> None:
        if self._profile is not None:
            return
        async with self._profile_lock:
            if self._profile is None:
                self._profile = await self.profile_loader.load(
                    self.user_id,
                    self.clone_id,
                )
                self.track.set_idle_image(self._profile.portrait_bytes)

    async def enqueue_reply(self, audio_bytes: bytes) -> None:
        await self.prepare()
        assert self._profile is not None
        video_bytes = await self.client.render(self._profile, audio_bytes)
        self.track.enqueue_encoded_video(video_bytes)


def create_ditto_video_session(
    *,
    user_id: str,
    clone_id: int,
    track: "QueuedVideoTrack",
) -> DittoVideoSession | None:
    config = DittoCallConfig.from_env()
    if config is None:
        print(
            "[DITTO_CALL] video renderer disabled: service is not configured",
            flush=True,
        )
        return None
    return DittoVideoSession(
        user_id=user_id,
        clone_id=clone_id,
        track=track,
        client=DittoRenderClient(config),
        profile_loader=FaceProfileLoader(),
    )


def _validate_profile(
    profile_bytes: bytes,
    user_id: str,
    clone_id: int,
) -> dict[str, Any]:
    try:
        profile = json.loads(profile_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DittoRealtimeError(f"Invalid face profile JSON: {exc}") from exc
    if str(profile.get("userUuid")) != str(user_id):
        raise DittoRealtimeError("Face profile userUuid does not match the call.")
    try:
        profile_clone_id = int(profile.get("cloneId"))
    except (TypeError, ValueError) as exc:
        raise DittoRealtimeError("Face profile cloneId is invalid.") from exc
    if profile_clone_id != clone_id:
        raise DittoRealtimeError("Face profile cloneId does not match the call.")
    engine = profile.get("engine") or {}
    if str(engine.get("name", "")).lower() != "ditto":
        raise DittoRealtimeError("Face profile is not configured for Ditto.")
    return profile


def _read_local_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DittoRealtimeError(f"{label} not found: {path}")
    if path.stat().st_size <= 0 or path.stat().st_size > max_bytes:
        raise DittoRealtimeError(
            f"{label} must contain 1 to {max_bytes} bytes."
        )
    return path.read_bytes()


def _read_s3_object(
    s3_client: Any,
    *,
    bucket: str,
    object_key: str,
    max_bytes: int,
    label: str,
) -> tuple[bytes, str | None]:
    response = s3_client.get_object(Bucket=bucket, Key=object_key)
    declared_size = response.get("ContentLength")
    if declared_size is not None and int(declared_size) > max_bytes:
        raise DittoRealtimeError(f"{label} exceeds {max_bytes} bytes.")
    body = response["Body"]
    try:
        content = body.read(max_bytes + 1)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if not content or len(content) > max_bytes:
        raise DittoRealtimeError(
            f"{label} must contain 1 to {max_bytes} bytes."
        )
    return content, response.get("ContentType")


def _s3_last_modified(item: dict[str, Any]) -> datetime:
    value = item.get("LastModified")
    if isinstance(value, datetime):
        return value
    return datetime.min.replace(tzinfo=timezone.utc)


def _create_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise DittoRealtimeError(
            "boto3 is required to load Ditto face profiles from S3."
        ) from exc
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    return boto3.client("s3", region_name=region or "ap-northeast-2")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DittoRealtimeError(f"{name} must be a boolean value.")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value else default
    except ValueError as exc:
        raise DittoRealtimeError(f"{name} must be a number.") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError as exc:
        raise DittoRealtimeError(f"{name} must be an integer.") from exc
