import asyncio
import sys
import unittest
from types import ModuleType
from unittest.mock import AsyncMock, Mock


try:
    import aiortc  # noqa: F401
except ImportError:
    aiortc = ModuleType("aiortc")

    class RTCPeerConnection:
        pass

    aiortc.RTCPeerConnection = RTCPeerConnection
    sys.modules["aiortc"] = aiortc

from model_calling.webrtc.session import (
    WebRTCSession,
    close_session,
    get_call_clone_id,
    get_call_media_type,
    get_call_user,
    get_call_context,
    register_call_context,
    save_session,
)
from model_calling.clients.backend_call_context import CallContext


def _context(call_id: int, user_uuid: str, clone_id: int, media_type: str):
    return CallContext(
        schemaVersion=1,
        callId=call_id,
        roomId=f"room-{call_id}",
        mediaType=media_type,
        status="READY",
        clone={
            "cloneId": clone_id,
            "userUuid": user_uuid,
            "persona": {},
            "voice": {
                "voiceProfileId": 1,
                "voiceTrainingJobId": 2,
                "provider": "ELEVENLABS",
                "voiceId": f"voice-{call_id}",
            },
        },
    )


class WebRTCSessionRegistryTests(unittest.TestCase):
    def test_close_cancels_and_awaits_in_flight_tasks(self) -> None:
        async def run() -> None:
            call_id = 912346
            cancelled_count = 0

            async def wait_forever() -> None:
                nonlocal cancelled_count
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled_count += 1

            peer = Mock()
            peer.close = AsyncMock()
            session = WebRTCSession(
                call_id=call_id,
                room_id="room-close",
                ai_signal_id="ai-close",
                caller_signal_id="caller-close",
                peer_connection=peer,
                clone_user_uuid="member-close",
                clone_id=7,
                output_track=Mock(),
                utterance_queue=asyncio.Queue(),
            )
            session.pipeline_task = asyncio.create_task(wait_forever())
            session.pipeline_start_task = asyncio.create_task(wait_forever())
            save_session(session)
            await asyncio.sleep(0)

            await close_session(call_id, reason="TEST_CLOSE")

            self.assertTrue(session.pipeline_task.cancelled())
            self.assertTrue(session.pipeline_start_task.cancelled())
            self.assertEqual(cancelled_count, 2)
            peer.close.assert_awaited_once()

        asyncio.run(run())

    def test_registers_and_clears_member_and_clone_together(self) -> None:
        call_id = 912345
        context = _context(
            call_id,
            "5f0154ef-7d83-4ae4-a724-ae591e1c985e",
            6,
            "VIDEO",
        )
        register_call_context(context)

        self.assertIs(get_call_context(call_id), context)
        self.assertEqual(
            get_call_user(call_id),
            "5f0154ef-7d83-4ae4-a724-ae591e1c985e",
        )
        self.assertEqual(get_call_clone_id(call_id), 6)
        self.assertEqual(get_call_media_type(call_id), "VIDEO")

        asyncio.run(close_session(call_id))

        self.assertIsNone(get_call_user(call_id))
        self.assertIsNone(get_call_clone_id(call_id))
        self.assertIsNone(get_call_media_type(call_id))

    def test_contexts_are_isolated_by_call_id(self) -> None:
        first = _context(
            912347,
            "5f0154ef-7d83-4ae4-a724-ae591e1c985e",
            6,
            "VOICE",
        )
        second = _context(
            912348,
            "66506c15-c42a-455f-8af7-a23f76c03bb2",
            9,
            "VIDEO",
        )
        register_call_context(first)
        register_call_context(second)
        try:
            self.assertIs(get_call_context(first.callId), first)
            self.assertIs(get_call_context(second.callId), second)
            self.assertEqual(get_call_clone_id(first.callId), 6)
            self.assertEqual(get_call_clone_id(second.callId), 9)
        finally:
            asyncio.run(close_session(first.callId))
            asyncio.run(close_session(second.callId))

    def test_conversation_history_is_isolated_per_session(self) -> None:
        first = WebRTCSession(
            call_id=1,
            room_id="room-1",
            ai_signal_id="ai-1",
            caller_signal_id="caller-1",
            peer_connection=Mock(),
            clone_user_uuid="member-1",
            clone_id=1,
            output_track=Mock(),
            utterance_queue=asyncio.Queue(),
        )
        second = WebRTCSession(
            call_id=2,
            room_id="room-2",
            ai_signal_id="ai-2",
            caller_signal_id="caller-2",
            peer_connection=Mock(),
            clone_user_uuid="member-1",
            clone_id=1,
            output_track=Mock(),
            utterance_queue=asyncio.Queue(),
        )

        first.conversation_history.append({"role": "user", "content": "안녕"})

        self.assertEqual(len(first.conversation_history), 1)
        self.assertEqual(second.conversation_history, [])


if __name__ == "__main__":
    unittest.main()
