import sys
import unittest
from types import SimpleNamespace

try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

from model_training.face_training.preview import infer_user_uuid
from model_training.face_training.worker import FaceTrainingWorkerError


class FaceTrainingPreviewTests(unittest.TestCase):
    def test_infers_same_user_from_multiple_object_keys(self) -> None:
        user_uuid = "65ebdde2-a48d-4a1c-b492-d59532a77557"

        inferred = infer_user_uuid(
            [
                f"face-videos/{user_uuid}/front.mov",
                f"face-videos/{user_uuid}/update.mp4",
            ]
        )

        self.assertEqual(inferred, user_uuid)

    def test_rejects_object_keys_for_different_users(self) -> None:
        with self.assertRaisesRegex(
            FaceTrainingWorkerError,
            "same user",
        ):
            infer_user_uuid(
                [
                    "face-videos/65ebdde2-a48d-4a1c-b492-d59532a77557/a.mov",
                    "face-videos/a22811d2-4103-48b7-8ff4-c04f51f2aed8/b.mov",
                ]
            )

    def test_rejects_key_without_user_uuid(self) -> None:
        with self.assertRaisesRegex(
            FaceTrainingWorkerError,
            "Unable to infer",
        ):
            infer_user_uuid(["uploads/face.mov"])


if __name__ == "__main__":
    unittest.main()
