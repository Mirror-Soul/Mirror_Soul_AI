import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from shared.clone_voice import (
    CloneVoiceMismatch,
    find_active_clone_voice,
)


USER_UUID = "65ebdde2-a48d-4a1c-b492-d59532a77557"


class _Cursor:
    def __init__(self, row: dict) -> None:
        self.row = row
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query: str, params: tuple[str]) -> None:
        self.params = params

    def fetchone(self) -> dict:
        return self.row


class _Connection:
    def __init__(self, row: dict) -> None:
        self.cursor_instance = _Cursor(row)

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    def close(self) -> None:
        return None


class CloneVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "clone_id": 6,
            "user_uuid": USER_UUID,
            "voice_training_job_id": 12,
            "elevenlabs_voice_id": "member-voice-id",
            "status": "ACTIVE",
            "is_active": 1,
        }

    def _modules(self) -> dict[str, ModuleType]:
        pymysql = ModuleType("pymysql")
        pymysql.connect = lambda **kwargs: _Connection(self.row)
        cursors = ModuleType("pymysql.cursors")
        cursors.DictCursor = object
        return {"pymysql": pymysql, "pymysql.cursors": cursors}

    def _environment(self) -> dict[str, str]:
        return {
            "DB_HOST": "db.example",
            "DB_PORT": "3306",
            "DB_USERNAME": "worker",
            "DB_PASSWORD": "secret",
            "DB_NAME": "mirrorsoul",
        }

    def test_resolves_voice_for_expected_member_clone(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            with patch.dict(sys.modules, self._modules()):
                result = find_active_clone_voice(
                    USER_UUID,
                    expected_clone_id=6,
                )

        self.assertEqual(result.clone_id, 6)
        self.assertEqual(result.user_uuid, USER_UUID)
        self.assertEqual(result.elevenlabs_voice_id, "member-voice-id")

    def test_rejects_face_and_voice_clone_mismatch(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            with patch.dict(sys.modules, self._modules()):
                with self.assertRaisesRegex(CloneVoiceMismatch, "different clones"):
                    find_active_clone_voice(
                        USER_UUID,
                        expected_clone_id=7,
                    )


if __name__ == "__main__":
    unittest.main()
