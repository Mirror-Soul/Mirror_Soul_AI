from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import numpy as np

from model_calling.realtime.trace import trace_fields
from model_calling.realtime.video_integrity import (
    VideoIntegrityConfig,
    VideoIntegrityError,
    audio_duration_seconds,
    validate_rendered_video,
)

if TYPE_CHECKING:
    from model_calling.realtime.video import QueuedVideoTrack


PROFILE_MAX_BYTES = 128 * 1024
PORTRAIT_MAX_BYTES = 10 * 1024 * 1024
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

_render_pool: _DittoRenderPool | None = None
_render_pool_loop: asyncio.AbstractEventLoop | None = None
_render_pool_urls: tuple[str, ...] = ()


class DittoRealtimeError(RuntimeError):
    def __init__(self, message: str, *, code: str = "DITTO_RENDER_FAILED") -> None:
        super().__init__(message)
        self.code = code


class _DittoRenderPool:
    def __init__(self, service_urls: tuple[str, ...]) -> None:
        self.service_urls = service_urls
        self._available: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        for index, service_url in enumerate(service_urls):
            self._available.put_nowait((index, service_url))
        self._admission_lock = asyncio.Lock()

    @property
    def worker_count(self) -> int:
        return len(self.service_urls)

    @property
    def all_busy(self) -> bool:
        return self._available.empty()

    async def acquire(
        self,
        timeout_seconds: float,
        *,
        slots: int = 1,
    ) -> tuple[tuple[int, str], ...]:
        if slots <= 0 or slots > self.worker_count:
            raise ValueError("Requested Ditto render slots are invalid.")
        deadline = time.monotonic() + timeout_seconds
        acquired: list[tuple[int, str]] = []
        try:
            await asyncio.wait_for(
                self._admission_lock.acquire(),
                timeout=_remaining_seconds(deadline),
            )
            try:
                for _ in range(slots):
                    acquired.append(
                        await asyncio.wait_for(
                            self._available.get(),
                            timeout=_remaining_seconds(deadline),
                        )
                    )
            finally:
                self._admission_lock.release()
            return tuple(acquired)
        except TimeoutError as exc:
            for worker in acquired:
                self._available.put_nowait(worker)
            raise DittoRealtimeError(
                "Ditto render queue wait exceeded "
                f"{timeout_seconds:.1f} seconds.",
                code="DITTO_RENDER_QUEUE_TIMEOUT",
            ) from exc
        except BaseException:
            for worker in acquired:
                self._available.put_nowait(worker)
            raise

    def release(self, workers: tuple[tuple[int, str], ...]) -> None:
        for worker in workers:
            self._available.put_nowait(worker)


DEFAULT_FADE_KEYS = "exp,pitch,yaw,roll,t"
IDLE_LOOP_SAMPLE_RATE = 16_000
IDLE_LOOP_CACHE_MAX_ENTRIES = 8
_idle_loop_cache: OrderedDict[str, bytes] = OrderedDict()


@dataclass(frozen=True)
class DittoMotionConfig:
    """Frames at the start/end of a clip that blend back to the portrait pose.

    Sent to the GPU service as ``fade_*`` form fields. A GPU service that
    predates these fields ignores them and renders exactly as before.
    """

    fade_in_frames: int = 0
    fade_out_frames: int = 0
    fade_keys: str = DEFAULT_FADE_KEYS

    @property
    def active(self) -> bool:
        return self.fade_in_frames > 0 or self.fade_out_frames > 0

    def form_fields(self) -> dict[str, str]:
        if not self.active:
            return {}
        return {
            "fade_in_frames": str(self.fade_in_frames),
            "fade_out_frames": str(self.fade_out_frames),
            "fade_type": "s",
            "fade_keys": self.fade_keys,
        }


@dataclass(frozen=True)
class DittoIdleLoopConfig:
    enabled: bool = False
    duration_seconds: float = 6.0
    fade_frames: int = 12

    def motion(self) -> DittoMotionConfig:
        return DittoMotionConfig(
            fade_in_frames=self.fade_frames,
            fade_out_frames=self.fade_frames,
        )


@dataclass(frozen=True)
class DittoCallConfig:
    service_url: str
    api_key: str
    service_urls: tuple[str, ...] = ()
    timeout_seconds: float = 300.0
    max_response_bytes: int = 100 * 1024 * 1024
    retry_attempts: int = 6
    retry_base_seconds: float = 1.0
    queue_timeout_seconds: float = 90.0
    parallel_max_audio_seconds: float = 8.0
    video_integrity: VideoIntegrityConfig = VideoIntegrityConfig()
    reply_motion: DittoMotionConfig = DittoMotionConfig()
    idle_loop: DittoIdleLoopConfig = DittoIdleLoopConfig()

    @property
    def worker_urls(self) -> tuple[str, ...]:
        return self.service_urls or (self.service_url,)

    @classmethod
    def from_env(cls) -> "DittoCallConfig | None":
        service_url = os.getenv("DITTO_CALL_SERVICE_URL", "").strip()
        service_urls_value = os.getenv("DITTO_CALL_SERVICE_URLS", "").strip()
        api_key = os.getenv("DITTO_CALL_SERVICE_API_KEY", "").strip()
        service_urls = _parse_service_urls(service_urls_value)
        if service_url:
            service_urls = _deduplicate((service_url, *service_urls))
        if not service_urls and not api_key:
            return None
        if not service_urls or not api_key:
            raise DittoRealtimeError(
                "A Ditto call service URL and DITTO_CALL_SERVICE_API_KEY "
                "must be configured together. Set DITTO_CALL_SERVICE_URL or "
                "DITTO_CALL_SERVICE_URLS."
            )

        service_urls = tuple(url.rstrip("/") for url in service_urls)
        for configured_url in service_urls:
            _validate_service_url(configured_url)

        timeout_seconds = _env_float("DITTO_CALL_TIMEOUT_SECONDS", 300.0)
        max_response_bytes = _env_int(
            "DITTO_CALL_MAX_RESPONSE_BYTES",
            100 * 1024 * 1024,
        )
        retry_attempts = _env_int("DITTO_CALL_RETRY_ATTEMPTS", 6)
        retry_base_seconds = _env_float(
            "DITTO_CALL_RETRY_BASE_SECONDS",
            1.0,
        )
        queue_timeout_seconds = _env_float(
            "DITTO_CALL_QUEUE_TIMEOUT_SECONDS",
            90.0,
        )
        parallel_max_audio_seconds = _env_float(
            "DITTO_CALL_PARALLEL_MAX_AUDIO_SECONDS",
            8.0,
        )
        if (
            timeout_seconds <= 0
            or max_response_bytes <= 0
            or retry_attempts <= 0
            or retry_base_seconds < 0
            or queue_timeout_seconds <= 0
            or parallel_max_audio_seconds < 0
        ):
            raise DittoRealtimeError(
                "Ditto call limits and retry settings are invalid."
            )
        reply_motion = DittoMotionConfig(
            fade_in_frames=_env_int("DITTO_CALL_REPLY_FADE_IN_FRAMES", 2),
            fade_out_frames=_env_int("DITTO_CALL_REPLY_FADE_OUT_FRAMES", 8),
            fade_keys=os.getenv("DITTO_CALL_FADE_KEYS", DEFAULT_FADE_KEYS).strip()
            or DEFAULT_FADE_KEYS,
        )
        idle_loop = DittoIdleLoopConfig(
            enabled=_env_bool("DITTO_CALL_IDLE_LOOP_ENABLED", True),
            duration_seconds=_env_float("DITTO_CALL_IDLE_LOOP_SECONDS", 6.0),
            fade_frames=_env_int("DITTO_CALL_IDLE_LOOP_FADE_FRAMES", 12),
        )
        if (
            not 0 <= reply_motion.fade_in_frames <= 50
            or not 0 <= reply_motion.fade_out_frames <= 50
            or not 2.0 <= idle_loop.duration_seconds <= 20.0
            or not 0 <= idle_loop.fade_frames <= 50
            or idle_loop.fade_frames * 2 >= idle_loop.duration_seconds * 25
        ):
            raise DittoRealtimeError(
                "Ditto fade or idle loop settings are invalid."
            )
        return cls(
            service_url=service_urls[0],
            api_key=api_key,
            service_urls=service_urls,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            retry_attempts=retry_attempts,
            retry_base_seconds=retry_base_seconds,
            queue_timeout_seconds=queue_timeout_seconds,
            parallel_max_audio_seconds=parallel_max_audio_seconds,
            video_integrity=VideoIntegrityConfig.from_env(),
            reply_motion=reply_motion,
            idle_loop=idle_loop,
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
        *,
        call_id: int | None = None,
        turn_id: int | None = None,
        audio_filename: str = "reply.mp3",
        audio_content_type: str = "audio/mpeg",
        motion: DittoMotionConfig | None = None,
    ) -> bytes:
        if not audio_bytes:
            raise DittoRealtimeError("Ditto reply audio must not be empty.")
        motion = self.config.reply_motion if motion is None else motion
        form_data = motion.form_fields()
        files = {
            "portrait": (
                profile.portrait_filename,
                profile.portrait_bytes,
                profile.portrait_content_type,
            ),
            "audio": (audio_filename, audio_bytes, audio_content_type),
        }
        if profile.profile_bytes is not None:
            files["profile"] = (
                "face-profile.json",
                profile.profile_bytes,
                "application/json",
            )

        render_pool = _get_render_pool(self.config.worker_urls)
        reserved_slots = 1
        expected_audio_duration = 0.0
        if (
            render_pool.worker_count > 1
            and self.config.parallel_max_audio_seconds > 0
        ):
            try:
                expected_audio_duration = await asyncio.to_thread(
                    audio_duration_seconds,
                    audio_bytes,
                )
            except VideoIntegrityError as exc:
                raise DittoRealtimeError(str(exc), code=exc.code) from exc
            if expected_audio_duration > self.config.parallel_max_audio_seconds:
                reserved_slots = render_pool.worker_count
        queued_at = time.monotonic()
        if render_pool.all_busy:
            print(
                "[DITTO_CALL] render queued behind another call: "
                f"{trace_fields(call_id, turn_id)} "
                f"workers={render_pool.worker_count}",
                flush=True,
            )
        response: httpx.Response | None = None
        last_transport_error: httpx.HTTPError | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            workers = await render_pool.acquire(
                self.config.queue_timeout_seconds,
                slots=reserved_slots,
            )
            worker_index, service_url = workers[0]
            queue_wait_seconds = time.monotonic() - queued_at
            print(
                "[DITTO_CALL] render slot acquired: "
                f"{trace_fields(call_id, turn_id)} "
                f"worker={worker_index + 1}/{render_pool.worker_count} "
                f"reserved_slots={reserved_slots} "
                f"audio_seconds={expected_audio_duration:.3f} "
                f"queue_wait_ms={round(queue_wait_seconds * 1000)} "
                f"fade_in={motion.fade_in_frames} "
                f"fade_out={motion.fade_out_frames}",
                flush=True,
            )
            try:
                if self._http_client is not None:
                    response = await self._send_once(
                        self._http_client,
                        service_url,
                        files,
                        form_data,
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self.config.timeout_seconds
                    ) as client:
                        response = await self._send_once(
                            client,
                            service_url,
                            files,
                            form_data,
                        )
                last_transport_error = None
            except httpx.HTTPError as exc:
                last_transport_error = exc
                response = None
            finally:
                render_pool.release(workers)

            retryable = (
                response is None
                or response.status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable or attempt >= self.config.retry_attempts:
                break
            delay = self.config.retry_base_seconds * (2 ** (attempt - 1))
            if response is None:
                print(
                    "[DITTO_CALL] transport error; retrying render: "
                    f"{trace_fields(call_id, turn_id)} attempt={attempt} "
                    f"delay={delay:.2f}s error={last_transport_error!r}",
                    flush=True,
                )
            else:
                print(
                    "[DITTO_CALL] renderer unavailable; retrying render: "
                    f"{trace_fields(call_id, turn_id)} "
                    f"status={response.status_code} attempt={attempt} "
                    f"delay={delay:.2f}s",
                    flush=True,
                )
            await asyncio.sleep(delay)

        if response is None:
            raise DittoRealtimeError(
                f"Ditto render request failed: {last_transport_error}"
            ) from last_transport_error

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
        try:
            integrity = await asyncio.to_thread(
                validate_rendered_video,
                response.content,
                audio_bytes,
                config=self.config.video_integrity,
            )
        except VideoIntegrityError as exc:
            raise DittoRealtimeError(str(exc), code=exc.code) from exc
        if self.config.video_integrity.enabled:
            print(
                "[DITTO_CALL] video integrity passed: "
                f"{trace_fields(call_id, turn_id)} "
                f"frames={integrity.frame_count} "
                f"duration={integrity.duration_seconds:.3f}s "
                f"audio_duration={integrity.expected_audio_duration_seconds:.3f}s "
                f"delta={integrity.duration_delta_seconds:.3f}s "
                f"fps={integrity.frame_rate:.3f} "
                f"resolution={integrity.width}x{integrity.height}",
                flush=True,
            )
        print(
            "[DITTO_CALL] render completed: "
            f"{trace_fields(call_id, turn_id)} bytes={len(response.content)} "
            f"seconds={response.headers.get('x-ditto-render-seconds', 'unknown')}",
            flush=True,
        )
        return response.content

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        service_url: str,
        files: dict[str, tuple[str, bytes, str]],
        form_data: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await client.post(
            f"{service_url}/api/v1/render",
            headers={"X-Ditto-Api-Key": self.config.api_key},
            files=files,
            data=form_data or None,
        )


class FaceProfileLoader:
    def __init__(self, *, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client

    async def load(
        self,
        user_id: str,
        clone_id: int,
        *,
        call_id: int | None = None,
    ) -> FaceRenderProfile:
        return await asyncio.to_thread(self._load, user_id, clone_id, call_id)

    def _load(
        self,
        user_id: str,
        clone_id: int,
        call_id: int | None = None,
    ) -> FaceRenderProfile:
        local_portrait = os.getenv("DITTO_CALL_LOCAL_PORTRAIT_PATH", "").strip()
        if local_portrait:
            return self._load_local(Path(local_portrait), user_id, clone_id)
        return self._load_s3(user_id, clone_id, call_id=call_id)

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

    def _load_s3(
        self,
        user_id: str,
        clone_id: int,
        *,
        call_id: int | None = None,
    ) -> FaceRenderProfile:
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
            f"{trace_fields(call_id)} user={user_id} "
            f"clone_id={clone_id} key={profile_key}",
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
        call_id: int | None = None,
        idle_loop: DittoIdleLoopConfig | None = None,
    ) -> None:
        self.idle_loop = idle_loop or DittoIdleLoopConfig()
        self.user_id = user_id
        self.clone_id = clone_id
        self.track = track
        self.client = client
        self.profile_loader = profile_loader
        self.call_id = call_id
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
                    call_id=self.call_id,
                )
                self.track.set_idle_image(self._profile.portrait_bytes)
                print(
                    "[DITTO_CALL] profile ready for call: "
                    f"{trace_fields(self.call_id)} user={self.user_id} "
                    f"clone_id={self.clone_id}",
                    flush=True,
                )

    async def prepare_idle_loop(self) -> bool:
        """Render a silent Ditto clip and loop it while the clone is idle.

        The clip starts and ends on the portrait's own pose (fade to source),
        so every reply, which also starts and ends there, joins it smoothly.
        Failure keeps the still portrait; the call itself is unaffected.
        """
        if not self.idle_loop.enabled:
            return False
        set_idle_video = getattr(self.track, "set_idle_video", None)
        if set_idle_video is None:
            return False
        await self.prepare()
        assert self._profile is not None
        cache_key = _idle_loop_cache_key(self._profile, self.idle_loop)
        started = time.monotonic()
        video_bytes = _idle_loop_cache.get(cache_key)
        source = "cache"
        if video_bytes is None:
            source = "render"
            print(
                "[DITTO_CALL] idle loop render started: "
                f"{trace_fields(self.call_id)} clone_id={self.clone_id} "
                f"seconds={self.idle_loop.duration_seconds:.1f} "
                f"fade_frames={self.idle_loop.fade_frames}",
                flush=True,
            )
            try:
                video_bytes = await self.client.render(
                    self._profile,
                    silent_wav_bytes(self.idle_loop.duration_seconds),
                    call_id=self.call_id,
                    audio_filename="idle.wav",
                    audio_content_type="audio/wav",
                    motion=self.idle_loop.motion(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    "[DITTO_CALL] idle loop unavailable; keeping still portrait: "
                    f"{trace_fields(self.call_id)} "
                    f"error_code={getattr(exc, 'code', 'IDLE_LOOP_FAILED')} "
                    f"error={exc!r}",
                    flush=True,
                )
                return False
        try:
            set_idle_video(video_bytes)
        except Exception as exc:
            print(
                "[DITTO_CALL] idle loop rejected; keeping still portrait: "
                f"{trace_fields(self.call_id)} error={exc!r}",
                flush=True,
            )
            _idle_loop_cache.pop(cache_key, None)
            return False
        _idle_loop_cache[cache_key] = video_bytes
        _idle_loop_cache.move_to_end(cache_key)
        while len(_idle_loop_cache) > IDLE_LOOP_CACHE_MAX_ENTRIES:
            _idle_loop_cache.popitem(last=False)
        print(
            "[DITTO_CALL] idle loop ready: "
            f"{trace_fields(self.call_id)} clone_id={self.clone_id} "
            f"source={source} bytes={len(video_bytes)} "
            f"elapsed_ms={round((time.monotonic() - started) * 1000)}",
            flush=True,
        )
        return True

    async def enqueue_reply(
        self,
        audio_bytes: bytes,
        *,
        turn_id: int | None = None,
    ) -> None:
        await self.prepare()
        assert self._profile is not None
        video_bytes = await self.client.render(
            self._profile,
            audio_bytes,
            call_id=self.call_id,
            turn_id=turn_id,
        )
        self.track.enqueue_encoded_video(video_bytes)


def create_ditto_video_session(
    *,
    call_id: int | None = None,
    user_id: str,
    clone_id: int,
    track: "QueuedVideoTrack",
) -> DittoVideoSession | None:
    config = DittoCallConfig.from_env()
    if config is None:
        print(
            "[DITTO_CALL] video renderer disabled: "
            f"{trace_fields(call_id)} service is not configured",
            flush=True,
        )
        return None
    return DittoVideoSession(
        user_id=user_id,
        clone_id=clone_id,
        track=track,
        client=DittoRenderClient(config),
        profile_loader=FaceProfileLoader(),
        call_id=call_id,
        idle_loop=config.idle_loop,
    )


def silent_wav_bytes(duration_seconds: float, *, seed: int = 7) -> bytes:
    """16 kHz mono PCM with a very low noise floor (about -66 dBFS).

    Pure digital silence can give speech-feature extractors degenerate
    input; a faint, deterministic noise floor keeps the mouth closed while
    Ditto still produces natural blinks and small head motion.
    """
    if duration_seconds <= 0:
        raise ValueError("Silent audio duration must be positive.")
    samples = max(1, round(duration_seconds * IDLE_LOOP_SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 16.0, samples).clip(-64, 64).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(IDLE_LOOP_SAMPLE_RATE)
        wav_file.writeframes(noise.tobytes())
    return output.getvalue()


def _idle_loop_cache_key(
    profile: FaceRenderProfile,
    config: DittoIdleLoopConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(profile.portrait_bytes)
    digest.update(b"\0")
    digest.update(profile.profile_bytes or b"")
    digest.update(
        f"\0{config.duration_seconds:.3f}:{config.fade_frames}".encode()
    )
    return digest.hexdigest()


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


def _get_render_pool(service_urls: tuple[str, ...]) -> _DittoRenderPool:
    global _render_pool, _render_pool_loop, _render_pool_urls
    loop = asyncio.get_running_loop()
    if (
        _render_pool is None
        or _render_pool_loop is not loop
        or _render_pool_urls != service_urls
    ):
        _render_pool = _DittoRenderPool(service_urls)
        _render_pool_loop = loop
        _render_pool_urls = service_urls
    return _render_pool


def _parse_service_urls(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return _deduplicate(
        tuple(item.strip() for item in value.split(",") if item.strip())
    )


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _validate_service_url(service_url: str) -> None:
    parsed = urlparse(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DittoRealtimeError("A configured Ditto call service URL is invalid.")
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


def _remaining_seconds(deadline: float) -> float:
    return max(0.001, deadline - time.monotonic())
