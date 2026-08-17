import json
import unittest

from model_training.face_training.message import (
    FaceTrainingMessageError,
    parse_face_training_message,
)


class FaceTrainingMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.message = {
            "schemaVersion": 1,
            "jobType": "FACE_PROFILE_BUILD",
            "jobId": 12,
            "source": "ONBOARDING_FACE",
            "userUuid": "16dc9bb9-e097-415f-9241-8dee558d858b",
            "cloneId": 3,
            "bucket": "mirror-soul-test",
            "objectKeys": [
                "face-videos/16dc9bb9-e097-415f-9241-8dee558d858b/input.mp4"
            ],
        }

    def test_parses_backend_contract(self) -> None:
        parsed = parse_face_training_message(json.dumps(self.message))

        self.assertEqual(parsed.schema_version, 1)
        self.assertEqual(parsed.job_type, "FACE_PROFILE_BUILD")
        self.assertEqual(parsed.job_id, 12)
        self.assertEqual(parsed.clone_id, 3)
        self.assertEqual(parsed.object_keys, self.message["objectKeys"])

    def test_rejects_unsupported_schema(self) -> None:
        self.message["schemaVersion"] = 2

        with self.assertRaisesRegex(
            FaceTrainingMessageError, "Unsupported schemaVersion"
        ):
            parse_face_training_message(json.dumps(self.message))

    def test_rejects_empty_object_keys(self) -> None:
        self.message["objectKeys"] = []

        with self.assertRaisesRegex(
            FaceTrainingMessageError, "non-empty list"
        ):
            parse_face_training_message(json.dumps(self.message))

    def test_rejects_path_traversal_object_key(self) -> None:
        self.message["objectKeys"] = ["face-videos/user/../secret.mp4"]

        with self.assertRaisesRegex(FaceTrainingMessageError, "Invalid objectKey"):
            parse_face_training_message(json.dumps(self.message))


if __name__ == "__main__":
    unittest.main()

