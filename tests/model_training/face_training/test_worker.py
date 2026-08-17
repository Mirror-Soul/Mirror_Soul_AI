import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

from model_training.face_training.worker import (
    FaceTrainingWorkerError,
    _download_face_video,
)


class _FakeS3Client:
    def __init__(self, content: bytes, content_type: str = "video/mp4") -> None:
        self.content = content
        self.content_type = content_type

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        return {
            "Body": io.BytesIO(self.content),
            "ContentLength": len(self.content),
            "ContentType": self.content_type,
        }


class FaceTrainingWorkerTests(unittest.TestCase):
    def test_downloads_video_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "input.mp4"
            downloaded = _download_face_video(
                _FakeS3Client(b"face-video"),
                bucket="test-bucket",
                object_key="face-videos/user/input.mp4",
                destination=destination,
            )

            self.assertEqual(destination.read_bytes(), b"face-video")
            self.assertEqual(downloaded.size_bytes, 10)
            self.assertEqual(downloaded.content_type, "video/mp4")

    def test_rejects_unsupported_content_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                FaceTrainingWorkerError, "Unsupported face video content type"
            ):
                _download_face_video(
                    _FakeS3Client(b"not-video", "application/octet-stream"),
                    bucket="test-bucket",
                    object_key="face-videos/user/input.bin",
                    destination=Path(temporary_directory) / "input.bin",
                )

    def test_removes_partial_file_when_size_limit_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "input.mp4"
            with patch.dict(
                os.environ,
                {"FACE_TRAINING_MAX_FILE_SIZE_BYTES": "4"},
            ):
                with self.assertRaisesRegex(FaceTrainingWorkerError, "exceeds"):
                    _download_face_video(
                        _FakeS3Client(b"12345"),
                        bucket="test-bucket",
                        object_key="face-videos/user/input.mp4",
                        destination=destination,
                    )

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
