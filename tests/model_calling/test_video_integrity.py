import io
import math
import unittest
import wave
from array import array
from fractions import Fraction

import av
import numpy as np

from model_calling.realtime.video_integrity import (
    VideoIntegrityConfig,
    VideoIntegrityError,
    validate_rendered_video,
)


def _wav_bytes(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    sample_count = int(duration_seconds * sample_rate)
    samples = array(
        "h",
        (
            int(7000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(sample_count)
        ),
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())
    return output.getvalue()


def _mp4_bytes(duration_seconds: float, fps: int = 25) -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    for index in range(round(duration_seconds * fps)):
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = index
        frame.time_base = Fraction(1, fps)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


class VideoIntegrityTests(unittest.TestCase):
    def test_accepts_decodable_video_matching_reply_audio(self) -> None:
        result = validate_rendered_video(
            _mp4_bytes(1.0),
            _wav_bytes(1.0),
            config=VideoIntegrityConfig(),
        )

        self.assertEqual(result.frame_count, 25)
        self.assertAlmostEqual(result.duration_seconds, 1.0, places=2)
        self.assertAlmostEqual(
            result.expected_audio_duration_seconds,
            1.0,
            places=2,
        )
        self.assertEqual((result.width, result.height), (32, 32))
        self.assertAlmostEqual(result.frame_rate, 25.0, places=2)

    def test_rejects_payload_without_video_stream(self) -> None:
        with self.assertRaises(VideoIntegrityError) as raised:
            validate_rendered_video(
                _wav_bytes(1.0),
                _wav_bytes(1.0),
                config=VideoIntegrityConfig(),
            )

        self.assertEqual(raised.exception.code, "DITTO_VIDEO_STREAM_MISSING")

    def test_rejects_video_duration_mismatch(self) -> None:
        with self.assertRaises(VideoIntegrityError) as raised:
            validate_rendered_video(
                _mp4_bytes(1.0),
                _wav_bytes(4.0),
                config=VideoIntegrityConfig(
                    max_duration_delta_seconds=0.5,
                    max_duration_delta_ratio=0.1,
                ),
            )

        self.assertEqual(raised.exception.code, "DITTO_VIDEO_DURATION_MISMATCH")

    def test_rejects_truncated_mp4(self) -> None:
        content = _mp4_bytes(1.0)
        with self.assertRaises(VideoIntegrityError) as raised:
            validate_rendered_video(
                content[: len(content) // 2],
                _wav_bytes(1.0),
                config=VideoIntegrityConfig(),
            )

        self.assertIn(
            raised.exception.code,
            {
                "DITTO_VIDEO_DECODE_FAILED",
                "DITTO_VIDEO_STREAM_MISSING",
                "DITTO_VIDEO_TOO_FEW_FRAMES",
            },
        )


if __name__ == "__main__":
    unittest.main()
