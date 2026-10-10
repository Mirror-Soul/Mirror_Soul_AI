import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fastapi.testclient import TestClient

from ditto_server.app import create_app
from ditto_server.config import DittoServiceConfig
from ditto_server.engine import DittoEngineBusyError, DittoRenderMetrics


class FakeEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.load_thread_id = None
        self.render_calls = []
        self.busy = False

    def load(self) -> None:
        self.load_thread_id = threading.get_ident()
        self.loaded = True

    def status(self) -> dict[str, object]:
        return {
            "loaded": self.loaded,
            "busy": self.busy,
            "gpu": "Test GPU",
            "modelLoadSeconds": 1.0,
            "renderCount": len(self.render_calls),
            "lastRenderSeconds": None,
            "lastError": None,
            "streamingAvailable": True,
        }

    def render(
        self,
        source_path,
        audio_path,
        output_path,
        *,
        settings,
        seed,
    ) -> DittoRenderMetrics:
        if self.busy:
            raise DittoEngineBusyError("Ditto GPU is already rendering.")
        self.render_calls.append(
            {
                "source": Path(source_path),
                "audio": Path(audio_path),
                "settings": settings,
                "seed": seed,
                "thread_id": threading.get_ident(),
            }
        )
        Path(output_path).write_bytes(b"rendered-video")
        return DittoRenderMetrics(
            duration_seconds=2.5,
            output_size_bytes=14,
        )

    def stream_frames(
        self,
        source_path,
        audio_path,
        output_path,
        *,
        settings,
        on_frame,
        seed,
        should_cancel,
        **kwargs,
    ):
        self.stream_kwargs = kwargs
        for value in (10, 20):
            on_frame(np.full((8, 8, 3), value, dtype=np.uint8))


class DittoServiceAppTests(unittest.TestCase):
    def _config(self, **overrides) -> DittoServiceConfig:
        values = {
            "api_key": "secret-key",
            "max_portrait_bytes": 1024,
            "max_audio_bytes": 2048,
        }
        values.update(overrides)
        return DittoServiceConfig(**values)

    def test_health_and_readiness_report_loaded_engine(self) -> None:
        engine = FakeEngine()
        with TestClient(create_app(self._config(), engine=engine)) as client:
            health = client.get("/health")
            ready = client.get("/ready")

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["engine"]["loaded"])
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")

    def test_render_requires_api_key(self) -> None:
        engine = FakeEngine()
        with TestClient(create_app(self._config(), engine=engine)) as client:
            response = client.post(
                "/api/v1/render",
                files={
                    "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                    "audio": ("speech.wav", b"audio", "audio/wav"),
                },
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(engine.render_calls, [])

    def test_render_uses_profile_settings_and_returns_video(self) -> None:
        engine = FakeEngine()
        profile = json.dumps(
            {
                "engine": {
                    "name": "ditto",
                    "renderSettings": {
                        "cropScale": 2.3,
                        "smoothingKernel": 5,
                        "samplingTimesteps": 50,
                    },
                }
            }
        ).encode("utf-8")
        with TestClient(create_app(self._config(), engine=engine)) as client:
            response = client.post(
                "/api/v1/render",
                headers={"X-Ditto-Api-Key": "secret-key"},
                files={
                    "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                    "audio": ("speech.wav", b"audio", "audio/wav"),
                    "profile": (
                        "face-profile.json",
                        profile,
                        "application/json",
                    ),
                },
                data={"seed": "2048"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"rendered-video")
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.headers["x-ditto-render-seconds"], "2.500")
        self.assertEqual(engine.render_calls[0]["settings"].smoothing_kernel, 5)
        self.assertEqual(engine.render_calls[0]["seed"], 2048)
        self.assertEqual(
            engine.render_calls[0]["thread_id"],
            engine.load_thread_id,
        )

    def test_render_rejects_oversized_upload(self) -> None:
        engine = FakeEngine()
        config = self._config(max_portrait_bytes=3)
        with TestClient(create_app(config, engine=engine)) as client:
            response = client.post(
                "/api/v1/render",
                headers={"X-Ditto-Api-Key": "secret-key"},
                files={
                    "portrait": ("portrait.jpg", b"large", "image/jpeg"),
                    "audio": ("speech.wav", b"audio", "audio/wav"),
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(engine.render_calls, [])

    def test_render_returns_429_when_gpu_is_busy(self) -> None:
        engine = FakeEngine()
        engine.busy = True
        with TestClient(create_app(self._config(), engine=engine)) as client:
            response = client.post(
                "/api/v1/render",
                headers={"X-Ditto-Api-Key": "secret-key"},
                files={
                    "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                    "audio": ("speech.wav", b"audio", "audio/wav"),
                },
            )

        self.assertEqual(response.status_code, 429)

    def test_stream_returns_length_prefixed_jpeg_frames(self) -> None:
        engine = FakeEngine()
        config = self._config(streaming_enabled=True)
        with patch(
            "ditto_server.app._encode_stream_frame",
            side_effect=[b"\xff\xd8frame-1", b"\xff\xd8frame-2"],
        ):
            with TestClient(create_app(config, engine=engine)) as client:
                response = client.post(
                    "/api/v1/render/stream",
                    headers={"X-Ditto-Api-Key": "secret-key"},
                    files={
                        "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                        "audio": ("speech.wav", b"audio", "audio/wav"),
                    },
                    data={"sampling_timesteps": "12"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith(
                "application/vnd.mirrorsoul.ditto-frame-stream"
            )
        )
        payload = response.content
        self.assertTrue(payload.startswith(b"MSDS1\n"))
        cursor = len(b"MSDS1\n")
        sizes = []
        while True:
            size = struct.unpack(">I", payload[cursor : cursor + 4])[0]
            cursor += 4
            if size == 0:
                break
            sizes.append(size)
            self.assertTrue(payload[cursor : cursor + size].startswith(b"\xff\xd8"))
            cursor += size
        self.assertEqual(len(sizes), 2)

    def test_stream_is_disabled_by_default(self) -> None:
        engine = FakeEngine()
        with TestClient(create_app(self._config(), engine=engine)) as client:
            response = client.post(
                "/api/v1/render/stream",
                headers={"X-Ditto-Api-Key": "secret-key"},
                files={
                    "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                    "audio": ("speech.wav", b"audio", "audio/wav"),
                },
            )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
