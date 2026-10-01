import asyncio
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from model_calling.realtime.ditto import (
    DittoCallConfig,
    DittoRealtimeError,
    DittoRenderClient,
    DittoVideoSession,
    FaceProfileLoader,
    FaceRenderProfile,
    _get_render_pool,
)
from model_calling.realtime.video_integrity import (
    VideoIntegrityConfig,
    VideoIntegrityError,
)


def _profile_bytes(user_id: str = "member-uuid", clone_id: int = 6) -> bytes:
    return json.dumps(
        {
            "userUuid": user_id,
            "cloneId": clone_id,
            "engine": {"name": "ditto"},
            "portrait": {
                "bucket": "face-bucket",
                "objectKey": "faces/portrait.jpg",
                "contentType": "image/jpeg",
            },
        }
    ).encode("utf-8")


class _Body(io.BytesIO):
    pass


class _Paginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kwargs):
        return self.pages


class _S3Client:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.paginator = _Paginator(
            [
                {
                    "Contents": [
                        {
                            "Key": "face-results/member-uuid/job-1/face-profile.json",
                            "LastModified": now - timedelta(days=1),
                        },
                        {
                            "Key": "face-results/member-uuid/job-2/face-profile.json",
                            "LastModified": now,
                        },
                    ]
                }
            ]
        )
        self.requested_keys = []

    def get_paginator(self, name):
        return self.paginator

    def get_object(self, *, Bucket, Key):
        self.requested_keys.append((Bucket, Key))
        content = _profile_bytes() if Key.endswith("face-profile.json") else b"jpg"
        return {
            "Body": _Body(content),
            "ContentLength": len(content),
            "ContentType": (
                "application/json"
                if Key.endswith("face-profile.json")
                else "image/jpeg"
            ),
        }


class _Track:
    def __init__(self) -> None:
        self.idle_images = []
        self.videos = []

    def set_idle_image(self, content):
        self.idle_images.append(content)

    def enqueue_encoded_video(self, content):
        self.videos.append(content)


class _ProfileLoader:
    def __init__(self, profile):
        self.profile = profile
        self.calls = 0

    async def load(self, user_id, clone_id, **kwargs):
        self.calls += 1
        return self.profile


class _RenderClient:
    def __init__(self):
        self.calls = []

    async def render(self, profile, audio_bytes, **kwargs):
        self.calls.append((profile, audio_bytes, kwargs))
        return b"mp4"


class DittoRealtimeTests(unittest.TestCase):
    def test_render_queue_timeout_has_stable_error_code(self) -> None:
        async def run() -> None:
            worker_urls = ("http://127.0.0.1:8080",)
            pool = _get_render_pool(worker_urls)
            worker = await pool.acquire(1.0)
            try:
                client = DittoRenderClient(
                    DittoCallConfig(
                        service_url=worker_urls[0],
                        api_key="secret",
                        queue_timeout_seconds=0.001,
                    ),
                    http_client=httpx.AsyncClient(
                        transport=httpx.MockTransport(
                            lambda request: httpx.Response(
                                200,
                                content=b"mp4",
                                headers={"content-type": "video/mp4"},
                            )
                        )
                    ),
                )
                profile = FaceRenderProfile(
                    portrait_bytes=b"portrait",
                    portrait_filename="portrait.jpg",
                    portrait_content_type="image/jpeg",
                )
                try:
                    with self.assertRaises(DittoRealtimeError) as raised:
                        await client.render(profile, b"audio", call_id=91)
                    self.assertEqual(
                        raised.exception.code,
                        "DITTO_RENDER_QUEUE_TIMEOUT",
                    )
                finally:
                    await client._http_client.aclose()
            finally:
                pool.release(worker)

        asyncio.run(run())

    def test_config_rejects_plain_http_to_remote_host(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DITTO_CALL_SERVICE_URL": "http://10.0.0.5:8080",
                "DITTO_CALL_SERVICE_URLS": "",
                "DITTO_CALL_SERVICE_API_KEY": "secret",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(DittoRealtimeError, "require HTTPS"):
                DittoCallConfig.from_env()

    def test_config_accepts_multiple_loopback_workers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DITTO_CALL_SERVICE_URL": "",
                "DITTO_CALL_SERVICE_URLS": (
                    "http://127.0.0.1:18080, http://127.0.0.1:18081"
                ),
                "DITTO_CALL_SERVICE_API_KEY": "secret",
            },
            clear=False,
        ):
            config = DittoCallConfig.from_env()

        assert config is not None
        self.assertEqual(
            config.worker_urls,
            (
                "http://127.0.0.1:18080",
                "http://127.0.0.1:18081",
            ),
        )

    def test_render_client_sends_profile_and_returns_mp4(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Ditto-Api-Key"], "secret")
            body = await request.aread()
            self.assertIn(b'filename="portrait.jpg"', body)
            self.assertIn(b'filename="face-profile.json"', body)
            self.assertIn(b'filename="reply.mp3"', body)
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> bytes:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http_client:
                client = DittoRenderClient(
                    DittoCallConfig(
                        service_url="http://127.0.0.1:8080",
                        api_key="secret",
                        video_integrity=VideoIntegrityConfig(enabled=False),
                    ),
                    http_client=http_client,
                )
                return await client.render(
                    FaceRenderProfile(
                        portrait_bytes=b"portrait",
                        portrait_filename="portrait.jpg",
                        portrait_content_type="image/jpeg",
                        profile_bytes=_profile_bytes(),
                    ),
                    b"audio",
                )

        self.assertEqual(asyncio.run(run()), b"rendered-mp4")

    def test_render_client_retries_busy_gpu(self) -> None:
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(429, json={"detail": "busy"})
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> bytes:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http_client:
                client = DittoRenderClient(
                    DittoCallConfig(
                        service_url="http://127.0.0.1:8080",
                        api_key="secret",
                        retry_attempts=3,
                        retry_base_seconds=0,
                        video_integrity=VideoIntegrityConfig(enabled=False),
                    ),
                    http_client=http_client,
                )
                return await client.render(
                    FaceRenderProfile(
                        portrait_bytes=b"portrait",
                        portrait_filename="portrait.jpg",
                        portrait_content_type="image/jpeg",
                    ),
                    b"audio",
                )

        self.assertEqual(asyncio.run(run()), b"rendered-mp4")
        self.assertEqual(attempts, 3)

    @patch(
        "model_calling.realtime.ditto.validate_rendered_video",
        side_effect=VideoIntegrityError(
            "duration mismatch",
            code="DITTO_VIDEO_DURATION_MISMATCH",
        ),
    )
    def test_render_client_preserves_video_integrity_error_code(
        self,
        _validate,
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> None:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http_client:
                client = DittoRenderClient(
                    DittoCallConfig(
                        service_url="http://127.0.0.1:8080",
                        api_key="secret",
                    ),
                    http_client=http_client,
                )
                with self.assertRaises(DittoRealtimeError) as raised:
                    await client.render(
                        FaceRenderProfile(
                            portrait_bytes=b"portrait",
                            portrait_filename="portrait.jpg",
                            portrait_content_type="image/jpeg",
                        ),
                        b"audio",
                    )
                self.assertEqual(
                    raised.exception.code,
                    "DITTO_VIDEO_DURATION_MISMATCH",
                )

        asyncio.run(run())

    def test_render_client_serializes_concurrent_callers(self) -> None:
        active_requests = 0
        max_active_requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.01)
            active_requests -= 1
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> list[bytes]:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http_client:
                config = DittoCallConfig(
                    service_url="http://127.0.0.1:8080",
                    api_key="secret",
                    video_integrity=VideoIntegrityConfig(enabled=False),
                )
                clients = [
                    DittoRenderClient(config, http_client=http_client),
                    DittoRenderClient(config, http_client=http_client),
                ]
                profile = FaceRenderProfile(
                    portrait_bytes=b"portrait",
                    portrait_filename="portrait.jpg",
                    portrait_content_type="image/jpeg",
                )
                return await asyncio.gather(
                    *(client.render(profile, b"audio") for client in clients)
                )

        self.assertEqual(
            asyncio.run(run()),
            [b"rendered-mp4", b"rendered-mp4"],
        )
        self.assertEqual(max_active_requests, 1)

    def test_render_client_uses_two_workers_concurrently(self) -> None:
        active_requests = 0
        max_active_requests = 0
        requested_ports = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active_requests, max_active_requests
            requested_ports.append(request.url.port)
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.02)
            active_requests -= 1
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> list[bytes]:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as http_client:
                config = DittoCallConfig(
                    service_url="http://127.0.0.1:18080",
                    service_urls=(
                        "http://127.0.0.1:18080",
                        "http://127.0.0.1:18081",
                    ),
                    api_key="secret",
                    parallel_max_audio_seconds=0,
                    video_integrity=VideoIntegrityConfig(enabled=False),
                )
                clients = [
                    DittoRenderClient(config, http_client=http_client)
                    for _ in range(3)
                ]
                profile = FaceRenderProfile(
                    portrait_bytes=b"portrait",
                    portrait_filename="portrait.jpg",
                    portrait_content_type="image/jpeg",
                )
                return await asyncio.gather(
                    *(client.render(profile, b"audio") for client in clients)
                )

        self.assertEqual(
            asyncio.run(run()),
            [b"rendered-mp4", b"rendered-mp4", b"rendered-mp4"],
        )
        self.assertEqual(max_active_requests, 2)
        self.assertEqual(set(requested_ports), {18080, 18081})

    def test_exclusive_render_reserves_every_worker(self) -> None:
        async def run() -> None:
            pool = _get_render_pool(
                (
                    "http://127.0.0.1:18080",
                    "http://127.0.0.1:18081",
                )
            )
            reservation = await pool.acquire(1.0, slots=2)
            try:
                with self.assertRaises(DittoRealtimeError) as raised:
                    await pool.acquire(0.001)
                self.assertEqual(
                    raised.exception.code,
                    "DITTO_RENDER_QUEUE_TIMEOUT",
                )
            finally:
                pool.release(reservation)

        asyncio.run(run())

    @patch(
        "model_calling.realtime.ditto.audio_duration_seconds",
        return_value=9.0,
    )
    def test_long_audio_serializes_across_worker_pool(self, _duration) -> None:
        active_requests = 0
        max_active_requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0.01)
            active_requests -= 1
            return httpx.Response(
                200,
                content=b"rendered-mp4",
                headers={"Content-Type": "video/mp4"},
            )

        async def run() -> None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as http_client:
                config = DittoCallConfig(
                    service_url="http://127.0.0.1:18080",
                    service_urls=(
                        "http://127.0.0.1:18080",
                        "http://127.0.0.1:18081",
                    ),
                    api_key="secret",
                    parallel_max_audio_seconds=8,
                    video_integrity=VideoIntegrityConfig(enabled=False),
                )
                clients = [
                    DittoRenderClient(config, http_client=http_client),
                    DittoRenderClient(config, http_client=http_client),
                ]
                profile = FaceRenderProfile(
                    portrait_bytes=b"portrait",
                    portrait_filename="portrait.jpg",
                    portrait_content_type="image/jpeg",
                )
                await asyncio.gather(
                    *(client.render(profile, b"audio") for client in clients)
                )

        asyncio.run(run())
        self.assertEqual(max_active_requests, 1)

    def test_local_profile_must_match_call_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portrait = root / "portrait.jpg"
            portrait.write_bytes(b"portrait")
            profile = root / "face-profile.json"
            profile.write_bytes(_profile_bytes(clone_id=99))
            with patch.dict(
                os.environ,
                {
                    "DITTO_CALL_LOCAL_PORTRAIT_PATH": str(portrait),
                    "DITTO_CALL_LOCAL_PROFILE_PATH": str(profile),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(DittoRealtimeError, "cloneId"):
                    asyncio.run(FaceProfileLoader().load("member-uuid", 6))

    def test_s3_loader_uses_latest_profile_and_its_portrait(self) -> None:
        s3_client = _S3Client()
        with patch.dict(
            os.environ,
            {
                "DITTO_CALL_LOCAL_PORTRAIT_PATH": "",
                "DITTO_CALL_LOCAL_PROFILE_PATH": "",
                "DITTO_CALL_S3_BUCKET": "profile-bucket",
                "DITTO_CALL_FACE_RESULT_PREFIX": "face-results",
            },
            clear=False,
        ):
            profile = asyncio.run(
                FaceProfileLoader(s3_client=s3_client).load("member-uuid", 6)
            )

        self.assertEqual(profile.portrait_bytes, b"jpg")
        self.assertEqual(
            s3_client.requested_keys,
            [
                (
                    "profile-bucket",
                    "face-results/member-uuid/job-2/face-profile.json",
                ),
                ("face-bucket", "faces/portrait.jpg"),
            ],
        )

    def test_video_session_loads_profile_once(self) -> None:
        profile = FaceRenderProfile(
            portrait_bytes=b"portrait",
            portrait_filename="portrait.jpg",
            portrait_content_type="image/jpeg",
        )
        loader = _ProfileLoader(profile)
        client = _RenderClient()
        track = _Track()
        session = DittoVideoSession(
            user_id="member-uuid",
            clone_id=6,
            track=track,
            client=client,
            profile_loader=loader,
            call_id=77,
        )

        async def run() -> None:
            await session.enqueue_reply(b"first")
            await session.enqueue_reply(b"second")

        asyncio.run(run())

        self.assertEqual(loader.calls, 1)
        self.assertEqual(track.idle_images, [b"portrait"])
        self.assertEqual(track.videos, [b"mp4", b"mp4"])
        self.assertEqual(client.calls[0][2]["call_id"], 77)


if __name__ == "__main__":
    unittest.main()
