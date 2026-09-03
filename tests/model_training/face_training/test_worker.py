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
    DownloadedFaceVideo,
    FaceTrainingWorkerError,
    _download_face_video,
    _evaluate_liveportrait_similarity,
    _select_liveportrait_inputs,
)
from model_training.face_training.face_similarity import (
    FaceSimilarityResult,
    FaceSimilarityUnavailable,
)
from model_training.face_training.liveportrait_runner import LivePortraitResult


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

    def test_selects_highest_quality_liveportrait_source_and_matching_video(
        self,
    ) -> None:
        first_source = Path("first-front.jpg")
        second_source = Path("second-front.jpg")
        videos = [
            DownloadedFaceVideo(
                bucket="bucket",
                object_key="first.mp4",
                content_type="video/mp4",
                local_path=Path("first.mp4"),
                size_bytes=1,
            ),
            DownloadedFaceVideo(
                bucket="bucket",
                object_key="second.mp4",
                content_type="video/mp4",
                local_path=Path("second.mp4"),
                size_bytes=1,
            ),
        ]
        selections = [
            SimpleNamespace(
                quality_gate_passed=True,
                selected_source_path=first_source,
                frames=[SimpleNamespace(path=first_source, quality_score=70.0)],
            ),
            SimpleNamespace(
                quality_gate_passed=True,
                selected_source_path=second_source,
                frames=[SimpleNamespace(path=second_source, quality_score=82.0)],
            ),
        ]

        source, driving = _select_liveportrait_inputs(videos, selections)

        self.assertEqual(source, second_source)
        self.assertEqual(driving, Path("second.mp4"))

    def test_rejects_liveportrait_when_quality_gate_does_not_pass(self) -> None:
        video = DownloadedFaceVideo(
            bucket="bucket",
            object_key="input.mp4",
            content_type="video/mp4",
            local_path=Path("input.mp4"),
            size_bytes=1,
        )
        selection = SimpleNamespace(
            quality_gate_passed=False,
            selected_source_path=None,
            frames=[],
        )

        with self.assertRaisesRegex(
            FaceTrainingWorkerError,
            "No face video passed",
        ):
            _select_liveportrait_inputs([video], [selection])

    def test_evaluates_liveportrait_against_all_accepted_frames(self) -> None:
        selected = SimpleNamespace(path=Path("front.jpg"), accepted=True)
        rejected = SimpleNamespace(path=Path("blurred.jpg"), accepted=False)
        selection = SimpleNamespace(frames=[selected, rejected])
        liveportrait = LivePortraitResult(
            source_path=Path("front.jpg"),
            driving_path=Path("driving.mp4"),
            output_path=Path("generated.mp4"),
            comparison_path=None,
            log_path=Path("liveportrait.log"),
            duration_seconds=1.0,
            command=("python", "inference.py"),
        )
        expected = FaceSimilarityResult(
            score=88.0,
            identity_score=90.0,
            render_quality_score=80.0,
            cosine_similarity=0.68,
            aligned_cosine_similarity=0.69,
            gallery_cosine_similarity=0.66,
            detection_rate=1.0,
            temporal_consistency=0.9,
            sharpness_retention=0.8,
            stability_factor=1.0,
            evaluated_frame_count=16,
            detected_frame_count=16,
            aligned_frame_count=16,
            reference_count=1,
            confidence="low",
            model_name="buffalo_l",
            provider="CPUExecutionProvider",
            calibration_version="provisional-v1",
            calibrated=False,
        )

        with patch.dict(os.environ, {"FACE_SIMILARITY_ENABLE": "true"}):
            with patch(
                "model_training.face_training.worker.evaluate_face_similarity",
                return_value=expected,
            ) as evaluate:
                actual = _evaluate_liveportrait_similarity(
                    [selection],
                    liveportrait,
                )

        self.assertIs(actual, expected)
        self.assertEqual(
            evaluate.call_args.kwargs["reference_images"],
            [Path("front.jpg")],
        )
        self.assertEqual(
            evaluate.call_args.kwargs["generated_video"],
            Path("generated.mp4"),
        )

    def test_optional_similarity_failure_keeps_face_output(self) -> None:
        selection = SimpleNamespace(
            frames=[SimpleNamespace(path=Path("front.jpg"), accepted=True)]
        )
        liveportrait = LivePortraitResult(
            source_path=Path("front.jpg"),
            driving_path=Path("driving.mp4"),
            output_path=Path("generated.mp4"),
            comparison_path=None,
            log_path=Path("liveportrait.log"),
            duration_seconds=1.0,
            command=("python", "inference.py"),
        )

        with patch.dict(
            os.environ,
            {
                "FACE_SIMILARITY_ENABLE": "true",
                "FACE_SIMILARITY_REQUIRED": "false",
            },
        ):
            with patch(
                "model_training.face_training.worker.evaluate_face_similarity",
                side_effect=FaceSimilarityUnavailable("model unavailable"),
            ):
                result = _evaluate_liveportrait_similarity(
                    [selection],
                    liveportrait,
                )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
