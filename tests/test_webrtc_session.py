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
    register_call_user,
    save_session,
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
        register_call_user(call_id, "member-uuid", 6, "VIDEO")

        self.assertEqual(get_call_user(call_id), "member-uuid")
        self.assertEqual(get_call_clone_id(call_id), 6)
        self.assertEqual(get_call_media_type(call_id), "VIDEO")

        asyncio.run(close_session(call_id))

        self.assertIsNone(get_call_user(call_id))
        self.assertIsNone(get_call_clone_id(call_id))
        self.assertIsNone(get_call_media_type(call_id))

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
