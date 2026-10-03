"""AI server must work without direct access to the backend MySQL database."""

import asyncio
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from model_calling.clients.backend_call_context import (
    CallContext,
    CallContextNotFound,
    CallContextUnavailable,
    fetch_call_context,
)
from model_calling.webrtc import session as session_module
from model_calling.webrtc.session import (
    call_context_count,
    close_session,
    get_call_context,
    prune_call_contexts,
    register_call_context,
)
from shared.config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_A = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"
USER_B = "66506c15-c42a-455f-8af7-a23f76c03bb2"
SECRET = "super-secret-internal-key"
VOICE_ID = "member-private-elevenlabs-voice-id"
RUNTIME_PACKAGES = ("model_calling", "model_training", "shared", "call_server", "ditto_server", "main.py")


def _context(call_id: int, user_uuid: str = USER_A, clone_id: int = 6) -> CallContext:
    return CallContext.model_validate(
        {
            "schemaVersion": 1,
            "callId": call_id,
            "roomId": f"room-{call_id}",
            "mediaType": "VOICE",
            "status": "READY",
            "clone": {
                "cloneId": clone_id,
                "userUuid": user_uuid,
                "persona": {"name": "회원"},
                "voice": {
                    "voiceProfileId": 1,
                    "voiceTrainingJobId": 2,
                    "provider": "ELEVENLABS",
                    "voiceId": VOICE_ID,
                },
            },
        }
    )


def _payload(call_id: int = 1) -> dict:
    return {
        "isSuccess": True,
        "code": "COMMON2000",
        "message": "ok",
        "result": _context(call_id).model_dump(mode="json"),
    }


class NoDatabaseDependencyTests(unittest.TestCase):
    def test_runtime_code_does_not_import_pymysql(self) -> None:
        offenders = []
        for package in RUNTIME_PACKAGES:
            root = REPO_ROOT / package
            files = [root] if root.is_file() else root.rglob("*.py")
            for path in files:
                text = path.read_text(encoding="utf-8")
                if "pymysql" in text.lower() or 'getenv("DB_' in text:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [])

    def test_requirements_do_not_include_pymysql(self) -> None:
        for path in REPO_ROOT.glob("requirements*.txt"):
            self.assertNotIn("pymysql", path.read_text(encoding="utf-8").lower(), path.name)

    def test_entrypoints_import_without_db_env_or_pymysql(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DB_")
        }
        with tempfile.TemporaryDirectory() as directory:
            env.update(
                {
                    "OPENAI_API_KEY": "test-only",
                    "PYTHONPATH": str(REPO_ROOT),
                    "RAG_DB_PATH": str(Path(directory) / "chroma"),
                }
            )
            code = (
                "import sys\n"
                "sys.modules['pymysql'] = None\n"
                "import main, call_server.main\n"
                "import model_training.face_training.worker\n"
                "import model_calling.voice_training.worker\n"
                "print('ok')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("ok", result.stdout)


class CallContextFetchTests(unittest.TestCase):
    def _fetch(self, handler, *, attempts: int = 2):
        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch.object(settings, "BACKEND_API_BASE_URL", "https://backend.example"), patch.object(
                    settings, "AI_INTERNAL_API_KEY", SECRET
                ), patch.object(settings, "BACKEND_CALL_CONTEXT_MAX_ATTEMPTS", attempts), patch.object(
                    settings, "BACKEND_CALL_CONTEXT_RETRY_BACKOFF_SECONDS", 0.0
                ):
                    return await fetch_call_context(1, client=client)

        return asyncio.run(run())

    def test_transient_timeout_is_retried_once(self) -> None:
        calls = []

        def handler(request: httpx.Request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, json=_payload())

        context = self._fetch(handler)
        self.assertEqual(context.callId, 1)
        self.assertEqual(len(calls), 2)

    def test_gateway_error_is_retried_then_reported_unavailable(self) -> None:
        calls = []

        def handler(request: httpx.Request):
            calls.append(request)
            return httpx.Response(502, json={"code": "BAD_GATEWAY"})

        with self.assertRaises(CallContextUnavailable):
            self._fetch(handler)
        self.assertEqual(len(calls), 2)

    def test_not_found_is_not_retried(self) -> None:
        calls = []

        def handler(request: httpx.Request):
            calls.append(request)
            return httpx.Response(404, json={"code": "CALL_4040", "message": "none"})

        with self.assertRaises(CallContextNotFound):
            self._fetch(handler)
        self.assertEqual(len(calls), 1)

    def test_retry_can_be_disabled(self) -> None:
        calls = []

        def handler(request: httpx.Request):
            calls.append(request)
            raise httpx.ConnectError("down", request=request)

        with self.assertRaises(CallContextUnavailable):
            self._fetch(handler, attempts=1)
        self.assertEqual(len(calls), 1)

    def test_secret_and_voice_id_are_not_logged_or_leaked(self) -> None:
        def failing(request: httpx.Request):
            raise httpx.ConnectError(f"down {request.headers.get('X-Internal-Api-Key')}", request=request)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            with self.assertRaises(CallContextUnavailable) as caught:
                self._fetch(failing)
            context = self._fetch(lambda request: httpx.Response(200, json=_payload()))
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertNotIn(SECRET, buffer.getvalue())
        self.assertNotIn(VOICE_ID, buffer.getvalue())
        self.assertNotIn(VOICE_ID, repr(caught.exception))
        self.assertEqual(context.clone.voice.voiceId, VOICE_ID)


class CallContextCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = [1000.0]
        self._patches = [
            patch.object(session_module, "_now", side_effect=lambda: self.clock[0]),
            patch.object(settings, "CALL_CONTEXT_CACHE_TTL_SECONDS", 60.0),
            patch.object(settings, "CALL_CONTEXT_CACHE_MAX_ENTRIES", 3),
        ]
        for item in self._patches:
            item.start()
        self.call_ids: list[int] = []

    def tearDown(self) -> None:
        for call_id in self.call_ids:
            session_module._sessions.pop(call_id, None)
            asyncio.run(close_session(call_id))
        for item in reversed(self._patches):
            item.stop()

    def _register(self, call_id: int, user_uuid: str = USER_A, clone_id: int = 6) -> CallContext:
        self.call_ids.append(call_id)
        context = _context(call_id, user_uuid, clone_id)
        register_call_context(context)
        return context

    def test_context_is_isolated_per_call_and_user(self) -> None:
        first = self._register(97001, USER_A, 6)
        second = self._register(97002, USER_B, 9)
        self.assertEqual(get_call_context(97001).clone.userUuid, USER_A)
        self.assertEqual(get_call_context(97002).clone.userUuid, USER_B)
        self.assertIsNot(get_call_context(97001), second)
        self.assertIs(get_call_context(97002), second)
        self.assertIsNot(first, second)

    def test_idle_context_expires_after_ttl(self) -> None:
        self._register(97011)
        self.clock[0] += 61
        self.assertIsNone(get_call_context(97011))

    def test_context_of_active_session_never_expires(self) -> None:
        context = self._register(97021)
        session_module._sessions[97021] = MagicMock()
        self.clock[0] += 10_000
        prune_call_contexts()
        self.assertIs(get_call_context(97021), context)

    def test_max_entries_evicts_oldest_idle_context_first(self) -> None:
        self._register(97031)
        session_module._sessions[97031] = MagicMock()
        for offset, call_id in enumerate((97032, 97033, 97034), start=1):
            self.clock[0] += offset
            self._register(call_id)
        before = call_context_count()
        self.assertLessEqual(before, 3 + len(_unrelated_contexts()))
        self.assertIsNotNone(get_call_context(97031))  # active session kept
        self.assertIsNone(get_call_context(97032))  # oldest idle evicted
        self.assertIsNotNone(get_call_context(97034))

    def test_call_end_removes_context(self) -> None:
        self._register(97041)
        asyncio.run(close_session(97041, reason="CALL_END"))
        self.assertIsNone(get_call_context(97041))


def _unrelated_contexts() -> list[int]:
    return [
        call_id
        for call_id in session_module._call_contexts
        if not 97000 <= call_id < 98000
    ]


class FacePreviewWithoutDatabaseTests(unittest.TestCase):
    def test_worker_skips_preview_when_no_voice_is_available(self) -> None:
        from model_training.face_training import worker

        message = MagicMock(user_uuid=USER_A, clone_id=6)
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "FACE_TRAINING_MEMBER_VOICE_PREVIEW_ENABLE": "true",
                "FACE_TRAINING_PREVIEW_FALLBACK_VOICE_ID": "",
                "FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH": "",
            },
        ), patch.object(worker, "_musetalk_config", return_value=MagicMock()), patch.object(
            worker, "_natural_motion_config", return_value=None
        ), patch(
            "model_training.face_training.member_voice_preview.run_musetalk_preview"
        ) as musetalk, contextlib.redirect_stdout(buffer):
            manifest = Path(directory) / "preprocess-manifest.json"
            manifest.write_text('{"videos": []}', encoding="utf-8")
            worker._maybe_generate_member_face_preview(manifest, message)

        musetalk.assert_not_called()
        self.assertIn("member voice face preview skipped", buffer.getvalue())

    def test_worker_preview_disabled_by_default(self) -> None:
        from model_training.face_training import worker

        with patch.dict(os.environ, {"FACE_TRAINING_MEMBER_VOICE_PREVIEW_ENABLE": "false"}), patch.object(
            worker, "generate_member_face_preview"
        ) as generate:
            worker._maybe_generate_member_face_preview(Path("unused"), MagicMock())
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
