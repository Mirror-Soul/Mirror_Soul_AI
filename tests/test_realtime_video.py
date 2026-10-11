import asyncio
import io
import unittest
from fractions import Fraction
from unittest.mock import patch

import av
import numpy as np

from model_calling.realtime.video import QueuedVideoTrack
from model_calling.realtime import video as video_module


def _encoded_video() -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=25)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    gradient = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    frame = av.VideoFrame.from_ndarray(
        np.stack([gradient, np.flipud(gradient), gradient], axis=2),
        format="rgb24",
    )
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


class QueuedVideoTrackTests(unittest.TestCase):
    def test_stream_frames_wait_for_activation_and_return_to_idle(self) -> None:
        async def run():
            track = QueuedVideoTrack(
                width=32,
                height=32,
                fps=1000,
                stream_buffer_max_frames=4,
            )
            frame = _encoded_video()
            track.set_idle_image(frame)
            stream_id = track.begin_frame_stream()
            await track.enqueue_stream_frame(stream_id, frame)
            await track.enqueue_stream_frame(stream_id, frame)
            buffered = await track.wait_for_stream_buffer(stream_id, 2, 1.0)
            track.activate_frame_stream(stream_id)
            track.synchronize_frame_stream()
            track.finish_frame_stream(stream_id)
            first = await track.recv()
            second = await track.recv()
            third = await track.recv()
            playing = track.is_playing
            track.stop()
            return buffered, first, second, third, playing

        buffered, first, second, third, playing = asyncio.run(run())

        self.assertEqual(buffered, 2)
        self.assertEqual((first.width, first.height), (32, 32))
        self.assertEqual((second.width, second.height), (32, 32))
        self.assertEqual((third.width, third.height), (32, 32))
        self.assertFalse(playing)

    def test_corrupt_stream_frame_falls_back_without_stopping_video(self) -> None:
        async def run():
            track = QueuedVideoTrack(width=32, height=32, fps=1000)
            track.set_idle_image(_encoded_video())
            stream_id = track.begin_frame_stream()
            await track.enqueue_stream_frame(stream_id, b"not-an-image")
            track.activate_frame_stream(stream_id)
            track.synchronize_frame_stream()
            track.finish_frame_stream(stream_id)
            first = await track.recv()
            second = await track.recv()
            track.stop()
            return first, second

        first, second = asyncio.run(run())

        self.assertEqual((first.width, first.height), (32, 32))
        self.assertEqual((second.width, second.height), (32, 32))

    def test_rejects_encoded_segment_without_decodable_frame(self) -> None:
        track = QueuedVideoTrack(width=32, height=32, fps=25)

        with self.assertRaises(Exception):
            track.enqueue_encoded_video(b"not-an-mp4")

        self.assertFalse(track.is_playing)
        track.stop()

    def test_sends_idle_frames_with_monotonic_rtp_timestamps(self) -> None:
        async def run():
            track = QueuedVideoTrack(width=32, height=32, fps=25)
            first = await track.recv()
            second = await track.recv()
            track.stop()
            return first, second

        first, second = asyncio.run(run())

        self.assertEqual((first.width, first.height), (32, 32))
        self.assertEqual(first.time_base, Fraction(1, 90_000))
        self.assertEqual(first.pts, 0)
        self.assertEqual(second.pts, 3_600)

    def test_plays_queued_segment_then_returns_to_idle(self) -> None:
        async def run():
            track = QueuedVideoTrack(width=32, height=32, fps=1000)
            video = _encoded_video()
            track.set_idle_image(video)
            track.enqueue_encoded_video(video)
            first = await track.recv()
            second = await track.recv()
            playing = track.is_playing
            track.stop()
            return first, second, playing

        first, second, playing = asyncio.run(run())

        self.assertEqual((first.width, first.height), (32, 32))
        self.assertEqual((second.width, second.height), (32, 32))
        self.assertFalse(playing)

    def test_idle_motion_changes_static_portrait_without_changing_dimensions(self) -> None:
        async def run():
            track = QueuedVideoTrack(
                width=32,
                height=32,
                fps=1000,
                idle_motion_scale=0.05,
                idle_motion_period_seconds=0.004,
            )
            track.set_idle_image(_encoded_video())
            first = await track.recv()
            second = await track.recv()
            track.stop()
            return (
                first.to_ndarray(format="rgb24"),
                second.to_ndarray(format="rgb24"),
            )

        first, second = asyncio.run(run())

        self.assertEqual(first.shape, (32, 32, 3))
        self.assertEqual(second.shape, (32, 32, 3))
        self.assertFalse(np.array_equal(first, second))


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class StreamCatchUpTests(unittest.TestCase):
    def _track(
        self,
        *,
        frames: int = 12,
        max_lag: int = 3,
        max_catch_up: int = 1,
    ) -> tuple[QueuedVideoTrack, _Clock]:
        clock = _Clock()
        encoded = _encoded_video()

        async def setup() -> QueuedVideoTrack:
            track = QueuedVideoTrack(
                width=32,
                height=32,
                fps=25,
                stream_buffer_max_frames=50,
                stream_max_lag_frames=max_lag,
                stream_max_catch_up_frames=max_catch_up,
            )
            track.set_idle_image(encoded)
            stream_id = track.begin_frame_stream()
            for _ in range(frames):
                await track.enqueue_stream_frame(stream_id, encoded)
            track.activate_frame_stream(stream_id)
            track.synchronize_frame_stream()
            return track

        with patch.object(video_module.time, "monotonic", clock):
            track = asyncio.run(setup())
        return track, clock

    def _pull(self, track: QueuedVideoTrack, clock: _Clock, at: float):
        clock.now = at
        with patch.object(video_module.time, "monotonic", clock):
            return track._next_stream_frame()

    def test_four_frame_lag_drops_only_one_excess_frame(self) -> None:
        track, clock = self._track(max_lag=3, max_catch_up=4)
        self._pull(track, clock, 100.00)
        self._pull(track, clock, 100.16)
        self.assertEqual(track._stream_dropped, 0)

        self._pull(track, clock, 100.25)
        self.assertEqual(track._stream_dropped, 1)

    def test_large_lag_is_limited_per_output_tick(self) -> None:
        track, clock = self._track(max_lag=3, max_catch_up=2)
        self._pull(track, clock, 100.00)
        self._pull(track, clock, 100.40)
        self.assertEqual(track._stream_dropped, 2)


if __name__ == "__main__":
    unittest.main()
