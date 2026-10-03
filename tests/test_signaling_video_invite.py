import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from model_calling.clients.backend_call_context import (
    CallContext,
    CallContextUnavailable,
)
from model_calling.signaling.handlers import handle_call_invite
from model_calling.webrtc.session import (
    close_session,
    get_call_clone_id,
    get_call_context,
    get_call_media_type,
    get_call_user,
)


USER_UUID = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"


def _context(call_id: int = 8001, room_id: str = "room") -> CallContext:
    return CallContext(
        schemaVersion=1,
        callId=call_id,
        roomId=room_id,
        mediaType="VIDEO",
        status="READY",
        clone={
            "cloneId": 6,
            "userUuid": USER_UUID,
            "persona": {},
            "voice": {
                "voiceProfileId": 32,
                "voiceTrainingJobId": 41,
                "provider": "ELEVENLABS",
                "voiceId": "backend-voice-id",
            },
        },
    )


def _invite(call_id=8001, room_id="room"):
    return {
        "type": "CALL_INVITE",
        "roomId": room_id,
        "from": "caller",
        "to": "ai",
        "data": {"callId": call_id},
    }


class _WebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send(self, message: str) -> None:
        self.messages.append(json.loads(message))


class SignalingVideoInviteTests(unittest.TestCase):
    def tearDown(self) -> None:
        for call_id in (8001, 8002, 8003, 8004):
            asyncio.run(close_session(call_id))

    def test_invite_fetches_and_registers_context_then_accepts_strict_shape(self) -> None:
        async def run():
            ws = _WebSocket()
            fetch = AsyncMock(return_value=_context())
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                fetch,
            ):
                await handle_call_invite(ws, _invite())
            return ws.messages, fetch

        messages, fetch = asyncio.run(run())

        fetch.assert_awaited_once_with(8001)
        self.assertEqual(
            messages[0],
            {
                "type": "CALL_ACCEPT",
                "roomId": "room",
                "from": "ai",
                "to": "caller",
                "data": {"callId": 8001},
            },
        )
        self.assertEqual(get_call_user(8001), USER_UUID)
        self.assertEqual(get_call_clone_id(8001), 6)
        self.assertEqual(get_call_media_type(8001), "VIDEO")

    def test_duplicate_invite_reuses_context_without_second_api_call(self) -> None:
        async def run():
            ws = _WebSocket()
            fetch = AsyncMock(return_value=_context())
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                fetch,
            ):
                await handle_call_invite(ws, _invite())
                await handle_call_invite(ws, _invite())
            return fetch.await_count

        self.assertEqual(asyncio.run(run()), 1)

    def test_room_mismatch_rejects_without_registration(self) -> None:
        async def run():
            ws = _WebSocket()
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                AsyncMock(return_value=_context(room_id="other-room")),
            ):
                await handle_call_invite(ws, _invite())
            return ws.messages

        messages = asyncio.run(run())
        self.assertEqual(messages[0]["data"]["reason"], "CALL_CONTEXT_MISMATCH")
        self.assertIsNone(get_call_context(8001))

    def test_invalid_call_id_is_rejected_before_api_call(self) -> None:
        async def run():
            ws = _WebSocket()
            fetch = AsyncMock()
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                fetch,
            ):
                await handle_call_invite(ws, _invite(call_id="8002"))
            return ws.messages, fetch.await_count

        messages, calls = asyncio.run(run())
        self.assertEqual(messages[0]["data"]["reason"], "INVALID_CALL_INVITE")
        self.assertEqual(calls, 0)

    def test_non_object_data_is_rejected_before_api_call(self) -> None:
        async def run():
            ws = _WebSocket()
            message = _invite()
            message["data"] = "invalid"
            fetch = AsyncMock()
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                fetch,
            ):
                await handle_call_invite(ws, message)
            return ws.messages, fetch.await_count

        messages, calls = asyncio.run(run())
        self.assertEqual(messages[0]["data"]["reason"], "INVALID_CALL_INVITE")
        self.assertEqual(calls, 0)

    def test_api_failure_rejects_without_registration(self) -> None:
        async def run():
            ws = _WebSocket()
            with patch(
                "model_calling.signaling.handlers.fetch_call_context",
                AsyncMock(
                    side_effect=CallContextUnavailable("backend unavailable")
                ),
            ):
                await handle_call_invite(ws, _invite(call_id=8003))
            return ws.messages

        messages = asyncio.run(run())
        self.assertEqual(
            messages[0]["data"]["reason"],
            "CALL_CONTEXT_UNAVAILABLE",
        )
        self.assertIsNone(get_call_context(8003))


if __name__ == "__main__":
    unittest.main()
