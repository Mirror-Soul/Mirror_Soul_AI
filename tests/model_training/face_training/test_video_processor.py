import unittest

from model_training.face_training.video_processor import (
    FaceVideoProcessingError,
    metadata_from_ffprobe_payload,
    validate_face_video_metadata,
)


class FaceVideoProcessorTests(unittest.TestCase):
    def test_parses_ffprobe_video_metadata(self) -> None:
        metadata = metadata_from_ffprobe_payload(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1080,
                        "height": 1920,
                        "avg_frame_rate": "30000/1001",
                    }
                ],
                "format": {"duration": "15.25", "format_name": "mov,mp4"},
            }
        )

        self.assertEqual(metadata.width, 1080)
        self.assertEqual(metadata.height, 1920)
        self.assertAlmostEqual(metadata.duration_seconds, 15.25)
        self.assertAlmostEqual(metadata.frame_rate, 29.970, places=3)

    def test_rejects_missing_video_stream(self) -> None:
        with self.assertRaisesRegex(FaceVideoProcessingError, "No video stream"):
            metadata_from_ffprobe_payload(
                {"streams": [{"codec_type": "audio"}], "format": {}}
            )

    def test_rejects_small_resolution(self) -> None:
        metadata = metadata_from_ffprobe_payload(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 192,
                        "height": 192,
                        "avg_frame_rate": "30/1",
                    }
                ],
                "format": {"duration": "10", "format_name": "mp4"},
            }
        )

        with self.assertRaisesRegex(FaceVideoProcessingError, "at least 256px"):
            validate_face_video_metadata(
                metadata,
                min_resolution=256,
                min_duration_seconds=1,
                max_duration_seconds=120,
            )


if __name__ == "__main__":
    unittest.main()

