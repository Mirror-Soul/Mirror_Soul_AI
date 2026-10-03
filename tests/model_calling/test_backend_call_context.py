import asyncio
import unittest
from unittest.mock import patch

import httpx

from model_calling.clients.backend_call_context import (
    CallContextConfigurationError,
    CallContextNotFound,
    CallContextUnavailable,
    CloneNotReady,
    InvalidCallContext,
    InvalidCallStatus,
    fetch_call_context,
)
from shared.config import settings


USER_UUID = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"


def _payload(**result_overrides):
    result = {
        "schemaVersion": 1,
        "callId": 1,
        "roomId": "call-abc",
        "mediaType": "VIDEO",
        "status": "READY",
        "clone": {
            "cloneId": 10,
            "userUuid": USER_UUID,
            "persona": {
                "name": "홍길동",
                "gender": "MALE",
                "birthDate": "2002-03-10",
                "job": "STUDENT",
                "jobDescription": "컴퓨터공학과 학생",
                "selfIntroduction": "영화와 음악을 좋아합니다.",
                "mbti": "INFP",
            },
            "voice": {
                "voiceProfileId": 32,
                "voiceTrainingJobId": 41,
                "provider": "ELEVENLABS",
                "voiceId": "elevenlabs-voice-id",
            },
        },
    }
    result.update(result_overrides)
    return {
        "isSuccess": True,
        "code": "COMMON2000",
        "message": "ok",
        "result": result,
        "error": None,
    }


class BackendCallContextClientTests(unittest.TestCase):
    def _run_with_handler(self, handler):
        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch.object(
                    settings,
                    "BACKEND_API_BASE_URL",
                    "https://backend.example/",
                ), patch.object(
                    settings,
                    "AI_INTERNAL_API_KEY",
                    "test-secret",
                ), patch.object(
                    settings,
                    "BACKEND_CALL_CONTEXT_TIMEOUT_SECONDS",
                    5.0,
                ):
                    return await fetch_call_context(1, client=client)

        return asyncio.run(run())

    def test_request_and_successful_response(self) -> None:
        def handler(request: httpx.Request):
            self.assertEqual(
                str(request.url),
                "https://backend.example/internal/ai/calls/1/context",
            )
            self.assertEqual(request.headers["X-Internal-Api-Key"], "test-secret")
            return httpx.Response(200, json=_payload())

        context = self._run_with_handler(handler)
        self.assertEqual(context.callId, 1)
        self.assertEqual(context.clone.userUuid, USER_UUID)
        self.assertEqual(context.clone.voice.voiceId, "elevenlabs-voice-id")

    def test_backend_error_mapping(self) -> None:
        cases = (
            (401, "INTERNAL_4010", CallContextConfigurationError),
            (503, "INTERNAL_5030", CallContextConfigurationError),
            (404, "CALL_4040", CallContextNotFound),
            (409, "CALL_4090", InvalidCallStatus),
            (409, "CLONE_4090", CloneNotReady),
            (409, "VOICE_4090", CloneNotReady),
            (500, "COMMON5000", CallContextUnavailable),
        )
        for status, code, error_type in cases:
            with self.subTest(status=status, code=code):
                def handler(request, status=status, code=code):
                    return httpx.Response(
                        status,
                        json={"isSuccess": False, "code": code, "message": code},
                    )

                with self.assertRaises(error_type):
                    self._run_with_handler(handler)

    def test_timeout_is_unavailable(self) -> None:
        def handler(request: httpx.Request):
            raise httpx.ReadTimeout("timed out", request=request)

        with self.assertRaises(CallContextUnavailable):
            self._run_with_handler(handler)

    def test_connection_failure_is_unavailable(self) -> None:
        def handler(request: httpx.Request):
            raise httpx.ConnectError("connection failed", request=request)

        with self.assertRaises(CallContextUnavailable):
            self._run_with_handler(handler)

    def test_invalid_json_is_rejected(self) -> None:
        with self.assertRaises(InvalidCallContext):
            self._run_with_handler(
                lambda request: httpx.Response(200, content=b"not-json")
            )

    def test_unsuccessful_wrapper_and_missing_result_are_rejected(self) -> None:
        for body in (
            {"isSuccess": False, "result": _payload()["result"]},
            {"isSuccess": True, "result": None},
        ):
            with self.subTest(body=body):
                with self.assertRaises(InvalidCallContext):
                    self._run_with_handler(
                        lambda request, body=body: httpx.Response(200, json=body)
                    )

    def test_schema_version_and_blank_voice_id_are_rejected(self) -> None:
        invalid_version = _payload(schemaVersion=2)
        blank_voice = _payload()
        blank_voice["result"]["clone"]["voice"]["voiceId"] = "  "
        for body in (invalid_version, blank_voice):
            with self.subTest(body=body):
                with self.assertRaises(InvalidCallContext):
                    self._run_with_handler(
                        lambda request, body=body: httpx.Response(200, json=body)
                    )

    def test_coerced_identifier_types_are_rejected(self) -> None:
        invalid_call_id = _payload(callId="1")
        invalid_clone_id = _payload()
        invalid_clone_id["result"]["clone"]["cloneId"] = True
        for body in (invalid_call_id, invalid_clone_id):
            with self.subTest(body=body):
                with self.assertRaises(InvalidCallContext):
                    self._run_with_handler(
                        lambda request, body=body: httpx.Response(200, json=body)
                    )


if __name__ == "__main__":
    unittest.main()
