import io
import json
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
    _handle_sqs_message,
    _select_liveportrait_inputs,
    run_worker,
)
from model_training.face_training.face_similarity import (
    FaceSimilarityResult,
    FaceSimilarityUnavailable,
)
from model_training.face_training.liveportrait_runner import LivePortraitResult
from model_training.face_training.profile_artifacts import FaceProfileArtifacts


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


class _FakeWorkerSqsClient:
    def __init__(self, message_body: str) -> None:
        self.message_body = message_body
        self.delete_calls = []

    def receive_message(self, **_kwargs):
        return {
            "Messages": [
                {
                    "Body": self.message_body,
                    "ReceiptHandle": "receipt-1",
                    "Attributes": {"ApproximateReceiveCount": "1"},
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.delete_calls.append(kwargs)


class FaceTrainingWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.message_body = json.dumps(
            {
                "schemaVersion": 1,
                "jobType": "FACE_PROFILE_BUILD",
                "jobId": 12,
                "source": "ONBOARDING_FACE",
                "userUuid": "16dc9bb9-e097-415f-9241-8dee558d858b",
                "cloneId": 3,
                "bucket": "mirror-soul-test",
                "objectKeys": ["face-videos/member/input.mp4"],
            }
        )

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

    def test_production_message_publishes_completion_before_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "preprocess-manifest.json"
            manifest_path.write_text(
                '{"videos": [{"frameSelection": '
                '{"qualityGatePassed": true}}], "faceSimilarity": null}',
                encoding="utf-8",
            )
            artifacts = FaceProfileArtifacts(
                bucket="mirror-soul-test",
                prefix="face-results/member/job-12",
                profile_key="face-results/member/job-12/face-profile.json",
                portrait_key="face-results/member/job-12/portrait.jpg",
                manifest_key=(
                    "face-results/member/job-12/preprocess-manifest.json"
                ),
                preview_key=None,
            )
            published = []

            def record_result(_client, *, queue_url, message):
                published.append((queue_url, message))
                return "message-id"

            with patch(
                "model_training.face_training.worker."
                "_preprocess_face_training_message",
                return_value=manifest_path,
            ), patch(
                "model_training.face_training.worker."
                "upload_face_profile_artifacts",
                return_value=artifacts,
            ), patch(
                "model_training.face_training.worker."
                "publish_face_training_result",
                side_effect=record_result,
            ):
                should_delete = _handle_sqs_message(
                    object(),
                    object(),
                    {
                        "Body": self.message_body,
                        "Attributes": {"ApproximateReceiveCount": "2"},
                    },
                    result_queue_url="https://sqs.example/results",
                )

        self.assertTrue(should_delete)
        self.assertEqual(
            [message.status for _, message in published],
            ["PROCESSING", "COMPLETED"],
        )
        self.assertEqual(published[-1][1].attempt_number, 2)
        self.assertEqual(
            published[-1][1].result["profileStatus"],
            "READY_FOR_RENDERING",
        )

    def test_failed_message_is_retained_for_retry(self) -> None:
        published = []

        def record_result(_client, *, queue_url, message):
            published.append(message)
            return "message-id"

        with patch(
            "model_training.face_training.worker."
            "_preprocess_face_training_message",
            side_effect=RuntimeError("render failed"),
        ), patch(
            "model_training.face_training.worker."
            "publish_face_training_result",
            side_effect=record_result,
        ), patch.dict(
            os.environ,
            {"FACE_TRAINING_DELETE_FAILED_MESSAGES": "false"},
        ):
            should_delete = _handle_sqs_message(
                object(),
                object(),
                {"Body": self.message_body},
                result_queue_url="https://sqs.example/results",
            )

        self.assertFalse(should_delete)
        self.assertEqual(
            [message.status for message in published],
            ["PROCESSING", "FAILED"],
        )
        self.assertEqual(published[-1].error["code"], "RuntimeError")

    def test_once_worker_deletes_only_completed_request(self) -> None:
        sqs_client = _FakeWorkerSqsClient(self.message_body)

        def client_for(service_name):
            return sqs_client if service_name == "sqs" else object()

        with patch.dict(
            os.environ,
            {
                "AWS_SQS_FACE_TRAINING_QUEUE_URL": (
                    "https://sqs.example/requests"
                ),
                "AWS_SQS_FACE_TRAINING_RESULT_QUEUE_URL": (
                    "https://sqs.example/results"
                ),
                "FACE_TRAINING_VISIBILITY_TIMEOUT": "30",
                "FACE_TRAINING_VISIBILITY_HEARTBEAT_SECONDS": "10",
            },
        ), patch(
            "model_training.face_training.worker._boto3_client",
            side_effect=client_for,
        ), patch(
            "model_training.face_training.worker._handle_sqs_message",
            return_value=True,
        ), patch(
            "model_training.face_training.worker._VisibilityHeartbeat"
        ):
            run_worker(once=True)

        self.assertEqual(
            sqs_client.delete_calls,
            [
                {
                    "QueueUrl": "https://sqs.example/requests",
                    "ReceiptHandle": "receipt-1",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
