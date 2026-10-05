"""Idle loop, fade-to-portrait and crossfade transitions for video calls."""

import asyncio
import io
import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import av
import httpx
import numpy as np
from fastapi.testclient import TestClient

from ditto_server.app import create_app
from ditto_server.config import DittoServiceConfig
from ditto_server.engine import (
    DittoEngine,
    DittoEngineError,
    DittoMotionOptions,
    DittoRenderMetrics,
    DittoRuntime,
    build_ditto_more_kwargs,
)
from model_calling.realtime import ditto as ditto_module
from model_calling.realtime.ditto import (
    DittoCallConfig,
    DittoIdleLoopConfig,
    DittoMotionConfig,
    DittoRealtimeError,
    DittoRenderClient,
    DittoVideoSession,
    FaceRenderProfile,
    silent_wav_bytes,
)
from model_calling.realtime.video import QueuedVideoTrack, QueuedVideoTrackError
from model_calling.realtime.video_integrity import VideoIntegrityConfig
from model_training.face_training.ditto_runner import DittoRenderSettings


SIZE = 32


def _clip(values: list[int]) -> bytes:
    """MP4 whose frame i is a flat gray image with brightness values[i]."""
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=25)
    stream.width = SIZE
    stream.height = SIZE
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}
    for value in values:
        frame = av.VideoFrame.from_ndarray(
            np.full((SIZE, SIZE, 3), value, dtype=np.uint8),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


def _brightness(frame: av.VideoFrame) -> float:
    return float(frame.to_ndarray(format="rgb24").mean())


def _collect(track: QueuedVideoTrack, count: int, before=None) -> list[float]:
    async def run() -> list[float]:
        if before is not None:
            before()
        return [_brightness(await track.recv()) for _ in range(count)]

    return asyncio.run(run())


PROFILE = FaceRenderProfile(
    portrait_bytes=b"portrait",
    portrait_filename="portrait.jpg",
    portrait_content_type="image/jpeg",
)


class CrossfadeTrackTests(unittest.TestCase):
    def test_idle_loop_replaces_still_portrait_and_loops(self) -> None:
        track = QueuedVideoTrack(
            width=SIZE, height=SIZE, fps=1000, idle_motion_enabled=False
        )
        track.set_idle_image(_clip([10]))
        track.set_idle_video(_clip([60, 80, 100]))

        values = _collect(track, 7)
        track.stop()

        self.assertTrue(track.has_idle_video)
        expected = [60, 80, 100, 60, 80, 100, 60]
        for value, target in zip(values, expected):
            self.assertAlmostEqual(value, target, delta=3)

    def test_reply_crossfades_in_and_idle_restarts_from_first_frame(self) -> None:
        track = QueuedVideoTrack(
            width=SIZE,
            height=SIZE,
            fps=1000,
            idle_motion_enabled=False,
            transition_frames=3,
        )
        track.set_idle_video(_clip([40, 40, 40, 40, 40, 90]))
        reply = _clip([200] * 6)

        async def run() -> list[float]:
            values = [_brightness(await track.recv()) for _ in range(2)]
            track.enqueue_encoded_video(reply)
            values += [_brightness(await track.recv()) for _ in range(10)]
            return values

        values = asyncio.run(run())
        track.stop()

        # idle, idle, then 3 blended reply frames rising toward 200
        self.assertAlmostEqual(values[0], 40, delta=3)
        self.assertAlmostEqual(values[2], 40 + (200 - 40) * 0.25, delta=4)
        self.assertAlmostEqual(values[3], 40 + (200 - 40) * 0.50, delta=4)
        self.assertAlmostEqual(values[4], 40 + (200 - 40) * 0.75, delta=4)
        self.assertAlmostEqual(values[5], 200, delta=3)
        self.assertAlmostEqual(values[7], 200, delta=3)
        # back to idle: fades from the last reply frame into the loop's
        # first frame (40), not wherever the loop had been before the reply.
        self.assertGreater(values[8], 40 + 10)
        self.assertLess(values[8], 200 - 10)
        self.assertAlmostEqual(values[11], 40, delta=3)
        self.assertEqual(track.transition_count, 2)

    def test_no_transition_frames_keeps_hard_cut(self) -> None:
        track = QueuedVideoTrack(
            width=SIZE, height=SIZE, fps=1000, idle_motion_enabled=False
        )
        track.set_idle_video(_clip([40, 40]))

        async def run() -> list[float]:
            values = [_brightness(await track.recv())]
            track.enqueue_encoded_video(_clip([200, 200]))
            values.append(_brightness(await track.recv()))
            return values

        values = asyncio.run(run())
        track.stop()
        self.assertAlmostEqual(values[1], 200, delta=3)
        self.assertEqual(track.transition_count, 0)

    def test_rejects_invalid_idle_video_and_keeps_portrait(self) -> None:
        track = QueuedVideoTrack(
            width=SIZE, height=SIZE, fps=1000, idle_motion_enabled=False
        )
        track.set_idle_image(_clip([120]))
        with self.assertRaises(Exception):
            track.set_idle_video(b"not-a-video")
        with self.assertRaises(QueuedVideoTrackError):
            track.set_idle_video(b"")
        values = _collect(track, 2)
        track.stop()
        self.assertFalse(track.has_idle_video)
        self.assertAlmostEqual(values[0], 120, delta=3)

    def test_transition_frames_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            QueuedVideoTrack(width=SIZE, height=SIZE, transition_frames=-1)
        with self.assertRaises(ValueError):
            QueuedVideoTrack(width=SIZE, height=SIZE, transition_frames=51)


class SilentAudioTests(unittest.TestCase):
    def test_silent_wav_is_16khz_mono_and_quiet(self) -> None:
        content = silent_wav_bytes(1.5)
        with wave.open(io.BytesIO(content)) as wav_file:
            self.assertEqual(wav_file.getframerate(), 16_000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getnframes(), 24_000)
            samples = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype="<i2"
            )
        self.assertLessEqual(int(np.abs(samples).max()), 64)
        self.assertEqual(content, silent_wav_bytes(1.5))


class RenderClientMotionTests(unittest.TestCase):
    def _render(self, config: DittoCallConfig, **kwargs) -> dict[str, bytes]:
        seen: dict[str, bytes] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = await request.aread()
            return httpx.Response(
                200, content=b"mp4", headers={"Content-Type": "video/mp4"}
            )

        async def run() -> None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as http_client:
                client = DittoRenderClient(config, http_client=http_client)
                await client.render(PROFILE, b"audio", **kwargs)

        asyncio.run(run())
        return seen

    def _config(self, **overrides) -> DittoCallConfig:
        values = {
            "service_url": "http://127.0.0.1:8080",
            "api_key": "secret",
            "video_integrity": VideoIntegrityConfig(enabled=False),
        }
        values.update(overrides)
        return DittoCallConfig(**values)

    def test_reply_sends_fade_fields(self) -> None:
        body = self._render(
            self._config(
                reply_motion=DittoMotionConfig(fade_in_frames=2, fade_out_frames=8)
            )
        )["body"]
        self.assertIn(b'name="fade_out_frames"\r\n\r\n8', body)
        self.assertIn(b'name="fade_in_frames"\r\n\r\n2', body)
        self.assertIn(b'name="fade_type"\r\n\r\ns', body)
        self.assertIn(b'filename="reply.mp3"', body)

    def test_inactive_motion_sends_no_fade_fields(self) -> None:
        body = self._render(self._config())["body"]
        self.assertNotIn(b"fade_", body)

    def test_idle_render_uses_wav_and_explicit_motion(self) -> None:
        body = self._render(
            self._config(),
            audio_filename="idle.wav",
            audio_content_type="audio/wav",
            motion=DittoMotionConfig(fade_in_frames=12, fade_out_frames=12),
        )["body"]
        self.assertIn(b'filename="idle.wav"', body)
        self.assertIn(b'name="fade_in_frames"\r\n\r\n12', body)


class _Track:
    def __init__(self, fail: bool = False) -> None:
        self.idle_images = []
        self.idle_videos = []
        self.fail = fail

    def set_idle_image(self, content):
        self.idle_images.append(content)

    def set_idle_video(self, content):
        if self.fail:
            raise QueuedVideoTrackError("bad clip")
        self.idle_videos.append(content)

    def enqueue_encoded_video(self, content):
        pass


class _Loader:
    async def load(self, user_id, clone_id, **kwargs):
        return PROFILE


class _Client:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error

    async def render(self, profile, audio_bytes, **kwargs):
        self.calls.append((audio_bytes, kwargs))
        if self.error is not None:
            raise self.error
        return b"idle-mp4"


class IdleLoopSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        ditto_module._idle_loop_cache.clear()

    def _session(self, track, client, enabled=True) -> DittoVideoSession:
        return DittoVideoSession(
            user_id="member-uuid",
            clone_id=6,
            track=track,
            client=client,
            profile_loader=_Loader(),
            call_id=77,
            idle_loop=DittoIdleLoopConfig(
                enabled=enabled, duration_seconds=4.0, fade_frames=10
            ),
        )

    def test_renders_silent_clip_with_fades_and_reuses_cache(self) -> None:
        client = _Client()
        first_track, second_track = _Track(), _Track()

        async def run():
            first = await self._session(first_track, client).prepare_idle_loop()
            second = await self._session(second_track, client).prepare_idle_loop()
            return first, second

        self.assertEqual(asyncio.run(run()), (True, True))
        self.assertEqual(len(client.calls), 1)
        audio_bytes, kwargs = client.calls[0]
        self.assertEqual(kwargs["audio_filename"], "idle.wav")
        self.assertEqual(kwargs["motion"].fade_in_frames, 10)
        self.assertEqual(kwargs["motion"].fade_out_frames, 10)
        self.assertEqual(kwargs["call_id"], 77)
        with wave.open(io.BytesIO(audio_bytes)) as wav_file:
            self.assertEqual(wav_file.getnframes(), 64_000)
        self.assertEqual(first_track.idle_videos, [b"idle-mp4"])
        self.assertEqual(second_track.idle_videos, [b"idle-mp4"])

    def test_render_failure_keeps_still_portrait(self) -> None:
        track = _Track()
        client = _Client(DittoRealtimeError("busy", code="DITTO_RENDER_QUEUE_TIMEOUT"))
        result = asyncio.run(self._session(track, client).prepare_idle_loop())
        self.assertFalse(result)
        self.assertEqual(track.idle_images, [b"portrait"])
        self.assertEqual(track.idle_videos, [])
        self.assertEqual(ditto_module._idle_loop_cache, {})

    def test_rejected_clip_is_not_cached(self) -> None:
        result = asyncio.run(
            self._session(_Track(fail=True), _Client()).prepare_idle_loop()
        )
        self.assertFalse(result)
        self.assertEqual(len(ditto_module._idle_loop_cache), 0)

    def test_disabled_idle_loop_does_not_render(self) -> None:
        client = _Client()
        result = asyncio.run(
            self._session(_Track(), client, enabled=False).prepare_idle_loop()
        )
        self.assertFalse(result)
        self.assertEqual(client.calls, [])


class CallConfigTests(unittest.TestCase):
    BASE = {
        "DITTO_CALL_SERVICE_URL": "http://127.0.0.1:8080",
        "DITTO_CALL_SERVICE_API_KEY": "secret",
    }

    def test_defaults_enable_reply_fade_and_idle_loop(self) -> None:
        with patch.dict(os.environ, self.BASE, clear=True):
            config = DittoCallConfig.from_env()
        self.assertEqual(config.reply_motion.fade_in_frames, 2)
        self.assertEqual(config.reply_motion.fade_out_frames, 8)
        self.assertTrue(config.idle_loop.enabled)
        self.assertEqual(config.idle_loop.duration_seconds, 6.0)
        self.assertEqual(config.idle_loop.fade_frames, 12)

    def test_can_turn_everything_off(self) -> None:
        env = dict(
            self.BASE,
            DITTO_CALL_REPLY_FADE_IN_FRAMES="0",
            DITTO_CALL_REPLY_FADE_OUT_FRAMES="0",
            DITTO_CALL_IDLE_LOOP_ENABLED="false",
        )
        with patch.dict(os.environ, env, clear=True):
            config = DittoCallConfig.from_env()
        self.assertFalse(config.reply_motion.active)
        self.assertEqual(config.reply_motion.form_fields(), {})
        self.assertFalse(config.idle_loop.enabled)

    def test_rejects_fades_longer_than_idle_clip(self) -> None:
        env = dict(
            self.BASE,
            DITTO_CALL_IDLE_LOOP_SECONDS="2",
            DITTO_CALL_IDLE_LOOP_FADE_FRAMES="30",
        )
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(DittoRealtimeError):
                DittoCallConfig.from_env()


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def load(self) -> None:
        pass

    def status(self):
        return {"loaded": True}

    def render(self, source, audio, output, *, settings, seed, **kwargs):
        self.calls.append(kwargs)
        Path(output).write_bytes(b"video")
        return DittoRenderMetrics(duration_seconds=1.0, output_size_bytes=5)


class DittoServiceMotionTests(unittest.TestCase):
    def _post(self, engine, data=None):
        config = DittoServiceConfig(api_key="key")
        with TestClient(create_app(config, engine=engine)) as client:
            return client.post(
                "/api/v1/render",
                headers={"X-Ditto-Api-Key": "key"},
                files={
                    "portrait": ("portrait.jpg", b"image", "image/jpeg"),
                    "audio": ("idle.wav", b"audio", "audio/wav"),
                },
                data=data or {},
            )

    def test_fade_fields_reach_engine(self) -> None:
        engine = _FakeEngine()
        response = self._post(
            engine,
            {"fade_in_frames": "12", "fade_out_frames": "12", "fade_type": "s"},
        )
        self.assertEqual(response.status_code, 200)
        motion = engine.calls[0]["motion"]
        self.assertEqual((motion.fade_in_frames, motion.fade_out_frames), (12, 12))
        self.assertEqual(motion.fade_keys, ("exp", "pitch", "yaw", "roll", "t"))

    def test_without_fade_fields_engine_is_called_as_before(self) -> None:
        engine = _FakeEngine()
        self.assertEqual(self._post(engine).status_code, 200)
        self.assertEqual(engine.calls, [{}])

    def test_invalid_fade_is_rejected(self) -> None:
        for data in (
            {"fade_out_frames": "-1"},
            {"fade_out_frames": "8", "fade_type": "x"},
            {"fade_out_frames": "8", "fade_keys": "exp,kp"},
        ):
            engine = _FakeEngine()
            response = self._post(engine, data)
            self.assertEqual(response.status_code, 422, data)
            self.assertEqual(engine.calls, [])


class DittoEngineMotionTests(unittest.TestCase):
    def test_more_kwargs_without_motion_match_previous_contract(self) -> None:
        self.assertEqual(
            build_ditto_more_kwargs(DittoRenderSettings()),
            {
                "setup_kwargs": {
                    "crop_scale": 2.3,
                    "smo_k_d": 5,
                    "sampling_timesteps": 50,
                }
            },
        )

    def test_more_kwargs_fade_to_source(self) -> None:
        kwargs = build_ditto_more_kwargs(
            DittoRenderSettings(),
            DittoMotionOptions(fade_in_frames=0, fade_out_frames=8),
        )
        self.assertEqual(kwargs["setup_kwargs"]["fade_type"], "s")
        self.assertEqual(
            kwargs["setup_kwargs"]["fade_out_keys"],
            ("exp", "pitch", "yaw", "roll", "t"),
        )
        self.assertEqual(kwargs["run_kwargs"]["fade_in"], -1)
        self.assertEqual(kwargs["run_kwargs"]["fade_out"], 8)
        other = build_ditto_more_kwargs(
            DittoRenderSettings(), DittoMotionOptions(fade_out_frames=8)
        )
        self.assertIsNot(kwargs["run_kwargs"]["ctrl_info"], other["run_kwargs"]["ctrl_info"])

    def test_engine_passes_motion_to_ditto_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints" / "ditto_cfg").mkdir(parents=True)
            (root / "checkpoints" / "ditto_trt_Ampere_Plus").mkdir()
            config = DittoServiceConfig(api_key="key", repository_dir=root)
            calls = []

            def run(sdk, audio_path, source_path, output_path, **kwargs):
                calls.append(kwargs["more_kwargs"])
                Path(output_path).write_bytes(b"video")

            runtime = DittoRuntime(
                sdk_factory=lambda cfg, data: object(),
                run=run,
                seed_everything=lambda seed: None,
                cuda_available=lambda: True,
                gpu_name=lambda: "GPU",
            )
            engine = DittoEngine(config, runtime_loader=lambda repo: runtime)
            with patch.object(DittoServiceConfig, "validate", lambda self: None):
                engine.load()
            source = root / "portrait.jpg"
            audio = root / "idle.wav"
            source.write_bytes(b"x")
            audio.write_bytes(b"x")
            engine.render(
                source,
                audio,
                root / "out.mp4",
                settings=DittoRenderSettings(),
                motion=DittoMotionOptions(fade_in_frames=12, fade_out_frames=12),
            )
            with self.assertRaises(DittoEngineError):
                engine.render(
                    source,
                    audio,
                    root / "bad.mp4",
                    settings=DittoRenderSettings(),
                    motion=DittoMotionOptions(fade_out_frames=999),
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["run_kwargs"]["fade_in"], 12)
        self.assertEqual(calls[0]["setup_kwargs"]["fade_type"], "s")


if __name__ == "__main__":
    unittest.main()


class IdleLoopMonitorTests(unittest.TestCase):
    def test_idle_loop_status_in_call_monitor(self) -> None:
        from tools import realtime_call_monitor as call_monitor

        ready = """
[SIGNALING] CALL_ACCEPT sent: callId=96 mediaType=VIDEO user=u clone_id=6
[WEBRTC] idle portrait ready: callId=96
[DITTO_CALL] idle loop render started: callId=96 clone_id=6 seconds=6.0 fade_frames=12
[DITTO_CALL] idle loop ready: callId=96 clone_id=6 source=render bytes=1 elapsed_ms=4100
"""
        snapshot = call_monitor.parse_call_logs(ready)
        self.assertEqual(snapshot.idle.status, "READY")
        self.assertEqual(snapshot.video.status, "READY")

        failed = """
[SIGNALING] CALL_ACCEPT sent: callId=97 mediaType=VIDEO user=u clone_id=6
[DITTO_CALL] idle loop unavailable; keeping still portrait: callId=97 error_code=DITTO_RENDER_FAILED error=x
"""
        snapshot = call_monitor.parse_call_logs(failed)
        self.assertEqual(snapshot.idle.status, "WARNING")
        self.assertNotEqual(snapshot.video.status, "FAILED")
