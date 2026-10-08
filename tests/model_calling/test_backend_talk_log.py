import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID

import httpx

from model_calling.clients.backend_talk_log import (
    MAX_MESSAGE_CHARS,
    TalkLogEntry,
    TalkLogSaveError,
    save_talk_log,
)
from shared.config import settings


EVENT_ID = UUID("8d3f1c52-5a8e-4f6e-9c1d-0d6a2f3c7b11")
STARTED = datetime(2026, 10, 8, 3, 0, 0, tzinfo=timezone.utc)


def _entry(**overrides):
    values = {
        "speaker": "CLONE",
        "message": "  안녕하세요, 반가워요.  ",
        "started_at": STARTED,
        "ended_at": STARTED + timedelta(seconds=2.5),
        "turn_id": 3,
        "event_id": EVENT_ID,
    }
    values.update(overrides)
    return TalkLogEntry(**values)


def _ok(talk_log_id=101, duplicated=False):
    return httpx.Response(
        200,
        json={
            "isSuccess": True,
            "code": "COMMON200",
            "message": "ok",
            "result": {
                "talkLogId": talk_log_id,
                "eventId": str(EVENT_ID),
                "duplicated": duplicated,
            },
        },
    )


def _error(status, code):
    return httpx.Response(
        status,
        json={"isSuccess": False, "code": code, "message": code, "result": None},
    )


class BackendTalkLogClientTests(unittest.TestCase):
    def _run(self, handler, entry=None, **setting_overrides):
        values = {
            "BACKEND_API_BASE_URL": "https://backend.example/",
            "AI_INTERNAL_API_KEY": "test-secret",
            "BACKEND_TALK_LOG_ENABLED": True,
            "BACKEND_TALK_LOG_TIMEOUT_SECONDS": 5.0,
            "BACKEND_TALK_LOG_MAX_ATTEMPTS": 3,
            "BACKEND_TALK_LOG_RETRY_BACKOFF_SECONDS": 0.0,
        }
        values.update(setting_overrides)

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                patches = [patch.object(settings, k, v) for k, v in values.items()]
                for item in patches:
                    item.start()
                try:
                    return await save_talk_log(77, entry or _entry(), client=client)
                finally:
                    for item in patches:
                        item.stop()

        return asyncio.run(run())

    def test_posts_contract_payload_with_internal_key(self):
        seen = {}

        def handler(request: httpx.Request):
            seen["url"] = str(request.url)
            seen["method"] = request.method
            seen["key"] = request.headers["X-Internal-Api-Key"]
            seen["body"] = json.loads(request.content)
            return _ok()

        result = self._run(handler)

        self.assertEqual(seen["method"], "POST")
        self.assertEqual(
            seen["url"],
            "https://backend.example/internal/ai/calls/77/talk-logs",
        )
        self.assertEqual(seen["key"], "test-secret")
        self.assertEqual(
            seen["body"],
            {
                "eventId": str(EVENT_ID),
                "speaker": "CLONE",
                "message": "안녕하세요, 반가워요.",
                "startedAt": "2026-10-08T03:00:00.000Z",
                "endedAt": "2026-10-08T03:00:02.500Z",
            },
        )
        self.assertEqual(result.talk_log_id, 101)
        self.assertFalse(result.duplicated)

    def test_duplicated_response_is_success(self):
        result = self._run(lambda request: _ok(talk_log_id=55, duplicated=True))
        self.assertTrue(result.duplicated)
        self.assertEqual(result.talk_log_id, 55)

    def test_message_is_trimmed_to_backend_limit(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _ok()

        self._run(handler, entry=_entry(message="가" * (MAX_MESSAGE_CHARS + 50)))
        self.assertEqual(len(bodies[0]["message"]), MAX_MESSAGE_CHARS)

    def test_naive_times_are_treated_as_utc_and_end_never_precedes_start(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _ok()

        naive = datetime(2026, 10, 8, 3, 0, 0)
        self._run(
            handler,
            entry=_entry(started_at=naive, ended_at=naive - timedelta(seconds=1)),
        )
        self.assertEqual(bodies[0]["startedAt"], "2026-10-08T03:00:00.000Z")
        self.assertEqual(bodies[0]["endedAt"], "2026-10-08T03:00:00.000Z")

    def test_missing_end_time_is_sent_as_null(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _ok()

        self._run(handler, entry=_entry(ended_at=None))
        self.assertIsNone(bodies[0]["endedAt"])

    def test_server_errors_are_retried_with_same_event_id(self):
        event_ids = []

        def handler(request):
            event_ids.append(json.loads(request.content)["eventId"])
            if len(event_ids) < 3:
                return _error(502, "COMMON502")
            return _ok()

        result = self._run(handler)
        self.assertEqual(result.talk_log_id, 101)
        self.assertEqual(event_ids, [str(EVENT_ID)] * 3)

    def test_transport_errors_are_retried_then_reported(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ConnectTimeout("timeout", request=request)

        with self.assertRaises(TalkLogSaveError) as raised:
            self._run(handler)
        self.assertEqual(calls, 3)
        self.assertEqual(raised.exception.code, "BACKEND_UNAVAILABLE")

    def test_business_errors_are_not_retried(self):
        cases = (
            (404, "CALL_4040"),
            (409, "TALK_LOG_4090"),
            (400, "TALK_LOG_4000"),
            (401, "INTERNAL_4010"),
            (503, "INTERNAL_5030"),
        )
        for status, code in cases:
            with self.subTest(code=code):
                calls = 0

                def handler(request, status=status, code=code):
                    nonlocal calls
                    calls += 1
                    return _error(status, code)

                with self.assertRaises(TalkLogSaveError) as raised:
                    self._run(handler)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(calls, 1)

    def test_missing_configuration_does_not_send(self):
        def handler(request):
            raise AssertionError("must not send")

        for overrides in (
            {"BACKEND_API_BASE_URL": ""},
            {"AI_INTERNAL_API_KEY": ""},
            {"BACKEND_TALK_LOG_ENABLED": False},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(TalkLogSaveError) as raised:
                    self._run(handler, **overrides)
                self.assertEqual(raised.exception.code, "AI_SERVER_CONFIG_ERROR")

    def test_blank_message_is_rejected_locally(self):
        def handler(request):
            raise AssertionError("must not send")

        with self.assertRaises(TalkLogSaveError) as raised:
            self._run(handler, entry=_entry(message="   "))
        self.assertEqual(raised.exception.code, "EMPTY_MESSAGE")


if __name__ == "__main__":
    unittest.main()
