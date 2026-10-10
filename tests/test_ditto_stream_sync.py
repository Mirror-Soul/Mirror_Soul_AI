"""Fade, colour and lip-sync behaviour of the Ditto frame stream."""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
import struct

from ditto_server.app import _encode_stream_frame, create_app
from ditto_server.config import DittoServiceConfig
from ditto_server.engine import (
    DittoEngine,
    DittoMotionOptions,
    DittoRuntime,
    _FrameCallbackWriter,
)
from model_calling.realtime import video as video_module
from model_calling.realtime.ditto import (
    DittoCallConfig,
    DittoMotionConfig,
    DittoRenderClient,
    FaceRenderProfile,
)
from model_calling.realtime.video import QueuedVideoTrack
from model_training.face_training.ditto_runner import DittoRenderSettings
from tests.ditto_server.test_app import FakeEngine
from tests.test_realtime_video import _encoded_video


RED_RGB = np.zeros((4, 4, 3), dtype=np.uint8)
RED_RGB[..., 0] = 255  # R in RGB order


class FrameColourTests(unittest.TestCase):
    def test_rgb_frames_from_ditto_reach_callback_as_bgr(self):
        received = []
        writer = _FrameCallbackWriter(lambda frame, fmt="bgr": None, received.append)
        writer(RED_RGB, fmt="rgb")
        self.assertEqual(received[0][0, 0].tolist(), [0, 0, 255])  # BGR red

    def test_bgr_frames_are_passed_through(self):
        received = []
        writer = _FrameCallbackWriter(lambda frame: None, received.append)
        bgr = RED_RGB[..., ::-1].copy()
        writer(bgr)
        self.assertEqual(received[0][0, 0].tolist(), [0, 0, 255])

    def test_jpeg_keeps_red_red(self):
        bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        decoded = cv2.imdecode(
            np.frombuffer(_encode_stream_frame(bgr, 90), np.uint8), cv2.IMREAD_COLOR
        )
        b, g, r = decoded[16, 16].tolist()
        self.assertGreater(r, 200)
        self.assertLess(b, 60)

    def test_jpeg_is_shrunk_to_fit_output_size_keeping_aspect(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        decoded = cv2.imdecode(
            np.frombuffer(_encode_stream_frame(frame, 80, max_size=(540, 960)), np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape[:2], (960, 540))
        small = np.zeros((100, 60, 3), dtype=np.uint8)
        decoded = cv2.imdecode(
            np.frombuffer(_encode_stream_frame(small, 80, max_size=(540, 960)), np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape[:2], (100, 60))  # never upscaled


class EngineFadeTests(unittest.TestCase):
    def _engine(self, directory, sdk):
        root = Path(directory)
        repository = root / "ditto"
        (repository / "checkpoints" / "ditto_pytorch").mkdir(parents=True)
        cfg = repository / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"
        cfg.parent.mkdir(parents=True)
        cfg.write_bytes(b"c")
        (root / "ffmpeg").mkdir()
        config = DittoServiceConfig(
            api_key="k", repository_dir=repository, ffmpeg_dir=root / "ffmpeg",
            streaming_enabled=True,
        )
        runtime = DittoRuntime(
            sdk_factory=lambda c, d: object(),
            run=lambda *a, **k: None,
            seed_everything=lambda seed: None,
            cuda_available=lambda: True,
            gpu_name=lambda: "GPU",
            online_sdk_factory=lambda c, d: sdk,
            # 3200 samples at 16 kHz = 0.2 s = 5 frames at 25 fps.
            load_audio=lambda path: np.zeros(3_200, dtype=np.float32),
        )
        engine = DittoEngine(config, runtime_loader=lambda repo: runtime)
        engine.load()
        for name in ("p.jpg", "a.wav"):
            (root / name).write_bytes(b"x")
        return engine, root / "p.jpg", root / "a.wav", root / "o.mp4"

    def _sdk(self, frames_per_chunk):
        class Writer:
            def __call__(self, frame, fmt="bgr"):
                return None

        class Sdk:
            def setup(self, source_path, output_path, **kwargs):
                self.setup_kwargs = kwargs
                self.writer = Writer()

            def setup_Nd(self, N_d, **kwargs):
                self.nd = (N_d, kwargs)

            def run_chunk(self, chunk, chunksize):
                for _ in range(frames_per_chunk):
                    self.writer(RED_RGB, fmt="rgb")

            def close(self):
                pass

        return Sdk()

    def test_fade_is_passed_to_online_sdk(self):
        sdk = self._sdk(frames_per_chunk=1)
        with tempfile.TemporaryDirectory() as directory:
            engine, source, audio, output = self._engine(directory, sdk)
            engine.stream_frames(
                source, audio, output,
                settings=DittoRenderSettings(sampling_timesteps=12),
                on_frame=lambda frame: None,
                motion=DittoMotionOptions(fade_in_frames=2, fade_out_frames=8),
            )
        self.assertEqual(sdk.setup_kwargs["fade_type"], "s")
        self.assertIn("pitch", sdk.setup_kwargs["fade_out_keys"])
        n_d, kwargs = sdk.nd
        self.assertEqual(n_d, 5)
        self.assertEqual((kwargs["fade_in"], kwargs["fade_out"]), (2, 8))
        self.assertEqual(kwargs["ctrl_info"], {})

    def test_without_motion_setup_is_unchanged(self):
        sdk = self._sdk(frames_per_chunk=1)
        with tempfile.TemporaryDirectory() as directory:
            engine, source, audio, output = self._engine(directory, sdk)
            engine.stream_frames(
                source, audio, output,
                settings=DittoRenderSettings(sampling_timesteps=12),
                on_frame=lambda frame: None,
            )
        self.assertNotIn("fade_type", sdk.setup_kwargs)
        self.assertEqual(sdk.nd, (5, {}))

    def test_frames_past_the_audio_are_trimmed(self):
        sdk = self._sdk(frames_per_chunk=4)  # 2 chunks x 4 = 8 frames > 5
        frames = []
        with tempfile.TemporaryDirectory() as directory:
            engine, source, audio, output = self._engine(directory, sdk)
            metrics = engine.stream_frames(
                source, audio, output,
                settings=DittoRenderSettings(sampling_timesteps=12),
                on_frame=frames.append,
            )
        self.assertEqual(len(frames), 5)
        self.assertEqual(metrics.frame_count, 5)
        self.assertEqual(frames[0][0, 0].tolist(), [0, 0, 255])


class StreamEndpointTests(unittest.TestCase):
    def _client(self, engine):
        from fastapi.testclient import TestClient

        config = DittoServiceConfig(
            api_key="secret-key",
            repository_dir=Path("."),
            ffmpeg_dir=Path("."),
            streaming_enabled=True,
        )
        return TestClient(create_app(config, engine=engine))

    def _post(self, client, data):
        return client.post(
            "/api/v1/render/stream",
            headers={"X-Ditto-Api-Key": "secret-key"},
            files={
                "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                "audio": ("speech.wav", b"audio", "audio/wav"),
            },
            data=data,
        )

    def test_fade_and_size_reach_engine_and_encoder(self):
        engine = FakeEngine()
        sizes = []

        def encode(frame, quality, *, max_size=None):
            sizes.append(max_size)
            return b"\xff\xd8x"

        with patch("ditto_server.app._encode_stream_frame", side_effect=encode):
            with self._client(engine) as client:
                response = self._post(client, {
                    "fade_in_frames": "2", "fade_out_frames": "8",
                    "max_width": "540", "max_height": "960",
                })
        self.assertEqual(response.status_code, 200)
        motion = engine.stream_kwargs["motion"]
        self.assertEqual((motion.fade_in_frames, motion.fade_out_frames), (2, 8))
        self.assertEqual(sizes, [(540, 960), (540, 960)])

    def test_old_requests_still_work(self):
        engine = FakeEngine()
        with patch("ditto_server.app._encode_stream_frame", return_value=b"\xff\xd8x"):
            with self._client(engine) as client:
                response = self._post(client, {})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(engine.stream_kwargs, {})

    def test_half_size_is_rejected(self):
        with self._client(FakeEngine()) as client:
            response = self._post(client, {"max_width": "540"})
        self.assertEqual(response.status_code, 422)


class StreamRequestTests(unittest.TestCase):
    def test_call_server_sends_fade_and_output_size(self):
        seen = {}
        payload = b"MSDS1\n" + struct.pack(">I", 3) + b"abc" + struct.pack(">I", 0)

        async def handler(request):
            seen["body"] = request.content.decode("latin-1")
            return httpx.Response(
                200, content=payload,
                headers={"Content-Type": "application/vnd.mirrorsoul.ditto-frame-stream"},
            )

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                client = DittoRenderClient(
                    DittoCallConfig(
                        service_url="http://127.0.0.1:18090",
                        api_key="secret",
                        reply_motion=DittoMotionConfig(fade_in_frames=2, fade_out_frames=8),
                        stream_max_width=540,
                        stream_max_height=960,
                    ),
                    http_client=http,
                )

                async def receive(frame):
                    return True

                await client.stream_frames(
                    FaceRenderProfile(
                        portrait_bytes=b"p", portrait_filename="p.jpg",
                        portrait_content_type="image/jpeg",
                    ),
                    b"audio",
                    on_frame=receive,
                )

        asyncio.run(run())
        body = seen["body"]
        for name, value in (
            ("fade_in_frames", "2"), ("fade_out_frames", "8"), ("fade_type", "s"),
            ("max_width", "540"), ("max_height", "960"),
        ):
            self.assertIn(f'name="{name}"\r\n\r\n{value}\r\n', body)

    def test_output_size_comes_from_realtime_video_env(self):
        env = {
            "DITTO_CALL_SERVICE_URL": "http://127.0.0.1:18080",
            "DITTO_CALL_SERVICE_API_KEY": "k",
            "REALTIME_VIDEO_WIDTH": "480",
            "REALTIME_VIDEO_HEIGHT": "854",
        }
        with patch.dict("os.environ", env, clear=False):
            config = DittoCallConfig.from_env()
        self.assertEqual((config.stream_max_width, config.stream_max_height), (480, 854))
        with patch.dict("os.environ", {**env, "DITTO_CALL_STREAM_MATCH_OUTPUT_SIZE": "false"}):
            config = DittoCallConfig.from_env()
        self.assertEqual((config.stream_max_width, config.stream_max_height), (0, 0))


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class LipSyncCatchUpTests(unittest.TestCase):
    def _track(self, frames, max_lag=3):
        clock = _Clock()
        frame = _encoded_video()

        async def setup():
            track = QueuedVideoTrack(
                width=32, height=32, fps=25,
                stream_buffer_max_frames=50, stream_max_lag_frames=max_lag,
            )
            track.set_idle_image(frame)
            stream_id = track.begin_frame_stream()
            for _ in range(frames):
                await track.enqueue_stream_frame(stream_id, frame)
            return track, stream_id

        with patch.object(video_module.time, "monotonic", clock):
            track, stream_id = asyncio.run(setup())
            track.activate_frame_stream(stream_id)
            track.synchronize_frame_stream()
        return track, clock

    def _pull(self, track, clock, at):
        clock.now = at
        with patch.object(video_module.time, "monotonic", clock):
            return track._next_stream_frame()

    def test_on_time_frames_are_never_skipped(self):
        track, clock = self._track(frames=6)
        for index in range(6):
            self.assertIsNotNone(self._pull(track, clock, 100.0 + index * 0.04))
        self.assertEqual(track._stream_dropped, 0)

    def test_late_frames_are_skipped_to_catch_up_with_audio(self):
        track, clock = self._track(frames=10)
        self._pull(track, clock, 100.00)
        self._pull(track, clock, 100.04)
        # GPU stalled: one second later frame 25 should be on screen, but
        # only frames 2..9 exist. Show the newest instead of frame 2.
        self.assertIsNotNone(self._pull(track, clock, 101.00))
        self.assertEqual(track._stream_dropped, 7)
        self.assertEqual(track._stream_consumed, 10)
        self.assertEqual(track._stream_max_lag, 23)

    def test_small_lag_is_tolerated(self):
        track, clock = self._track(frames=10)
        self._pull(track, clock, 100.00)
        self._pull(track, clock, 100.12)  # 3 frames late: within tolerance
        self.assertEqual(track._stream_dropped, 0)

    def test_catch_up_can_be_disabled(self):
        track, clock = self._track(frames=10, max_lag=0)
        self._pull(track, clock, 101.00)
        self.assertEqual(track._stream_dropped, 0)

    def test_empty_buffer_counts_as_stall(self):
        track, clock = self._track(frames=1)
        self._pull(track, clock, 100.00)
        self._pull(track, clock, 100.04)
        self._pull(track, clock, 100.08)
        self.assertEqual(track._stream_stalled, 2)

    def test_activation_does_not_start_clock_before_audio_is_queued(self):
        track, clock = self._track(frames=6)
        track._stream_activated_at = None

        self.assertIsNone(self._pull(track, clock, 101.00))
        self.assertEqual(track._stream_consumed, 0)
        self.assertEqual(track._stream_dropped, 0)

        with patch.object(video_module.time, "monotonic", clock):
            self.assertTrue(track.synchronize_frame_stream())
        self.assertIsNotNone(self._pull(track, clock, 101.00))
        self.assertEqual(track._stream_consumed, 1)
        self.assertEqual(track._stream_dropped, 0)

    def test_audio_queue_delay_holds_first_mouth_frame(self):
        track, clock = self._track(frames=6)
        with patch.object(video_module.time, "monotonic", clock):
            track.synchronize_frame_stream(0.20)

        self.assertIsNone(self._pull(track, clock, 100.19))
        self.assertEqual(track._stream_consumed, 0)
        self.assertIsNotNone(self._pull(track, clock, 100.20))
        self.assertEqual(track._stream_consumed, 1)
        self.assertEqual(track._stream_dropped, 0)


if __name__ == "__main__":
    unittest.main()
