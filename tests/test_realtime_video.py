import asyncio
import io
import unittest
from fractions import Fraction

import av
import numpy as np

from model_calling.realtime.video import QueuedVideoTrack


def _encoded_video() -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=25)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(
        np.full((32, 32, 3), 180, dtype=np.uint8),
        format="rgb24",
    )
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


class QueuedVideoTrackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
