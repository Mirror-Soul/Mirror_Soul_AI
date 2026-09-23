import json
import tempfile
import unittest
from pathlib import Path

from model_training.face_training.message import FaceTrainingMessage
from model_training.face_training.profile_artifacts import (
    FaceProfileArtifactError,
    upload_face_profile_artifacts,
)


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, ContentType):
        content = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[(Bucket, Key)] = {
            "body": content,
            "content_type": ContentType,
        }
        return {"ETag": '"etag"'}


class FaceProfileArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.message = FaceTrainingMessage(
            schema_version=1,
            job_type="FACE_PROFILE_BUILD",
            job_id=12,
            source="ONBOARDING_FACE",
            user_uuid="16dc9bb9-e097-415f-9241-8dee558d858b",
            clone_id=3,
            bucket="mirror-soul-test",
            object_keys=["face-videos/member/input.mp4"],
        )

    def test_uploads_portable_profile_and_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portrait = root / "front.jpg"
            portrait.write_bytes(b"portrait")
            manifest_path = root / "preprocess-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "frameSelection": {
                                    "qualityGatePassed": True,
                                    "qualityTier": "LOW",
                                    "selectionMode": "BEST_EFFORT",
                                    "qualityWarnings": ["low_sharpness"],
                                    "selectedSourcePath": str(portrait),
                                    "frames": [
                                        {
                                            "path": str(portrait),
                                            "qualityScore": 88.0,
                                            "sharpness": 72.0,
                                            "view": "front",
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            client = _FakeS3Client()

            result = upload_face_profile_artifacts(
                client,
                message=self.message,
                manifest_path=manifest_path,
            )

            expected_prefix = (
                "face-results/16dc9bb9-e097-415f-9241-8dee558d858b/job-12"
            )
            self.assertEqual(result.prefix, expected_prefix)
            self.assertIsNone(result.preview_key)
            profile = json.loads(
                client.objects[(self.message.bucket, result.profile_key)][
                    "body"
                ].decode("utf-8")
            )
            self.assertEqual(profile["engine"]["name"], "ditto")
            self.assertEqual(
                profile["engine"]["renderSettings"]["smoothingKernel"],
                5,
            )
            self.assertEqual(profile["portrait"]["objectKey"], result.portrait_key)
            self.assertEqual(profile["quality"]["qualityScore"], 88.0)
            self.assertEqual(profile["quality"]["qualityTier"], "LOW")
            self.assertEqual(
                profile["quality"]["selectionMode"],
                "BEST_EFFORT",
            )
            self.assertEqual(
                profile["quality"]["qualityWarnings"],
                ["low_sharpness"],
            )

    def test_uploads_existing_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portrait = root / "front.jpg"
            portrait.write_bytes(b"portrait")
            preview = root / "preview.mp4"
            preview.write_bytes(b"preview")
            manifest_path = root / "preprocess-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "frameSelection": {
                                    "qualityGatePassed": True,
                                    "selectedSourcePath": str(portrait),
                                    "frames": [{"path": str(portrait)}],
                                }
                            }
                        ],
                        "livePortrait": {"outputPath": str(preview)},
                    }
                ),
                encoding="utf-8",
            )
            client = _FakeS3Client()

            result = upload_face_profile_artifacts(
                client,
                message=self.message,
                manifest_path=manifest_path,
            )

            self.assertIsNotNone(result.preview_key)
            uploaded = client.objects[(self.message.bucket, result.preview_key)]
            self.assertEqual(uploaded["body"], b"preview")
            self.assertEqual(uploaded["content_type"], "video/mp4")

    def test_rejects_unsafe_result_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "preprocess-manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                FaceProfileArtifactError,
                "Invalid face result prefix",
            ):
                upload_face_profile_artifacts(
                    _FakeS3Client(),
                    message=self.message,
                    manifest_path=manifest_path,
                    result_prefix="../private",
                )

    def test_rejects_even_smoothing_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "preprocess-manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                FaceProfileArtifactError,
                "positive odd integer",
            ):
                upload_face_profile_artifacts(
                    _FakeS3Client(),
                    message=self.message,
                    manifest_path=manifest_path,
                    smoothing_kernel=4,
                )


if __name__ == "__main__":
    unittest.main()
