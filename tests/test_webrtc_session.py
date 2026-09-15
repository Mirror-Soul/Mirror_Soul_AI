import asyncio
import sys
import unittest
from types import ModuleType


try:
    import aiortc  # noqa: F401
except ImportError:
    aiortc = ModuleType("aiortc")

    class RTCPeerConnection:
        pass

    aiortc.RTCPeerConnection = RTCPeerConnection
    sys.modules["aiortc"] = aiortc

from model_calling.webrtc.session import (
    close_session,
    get_call_clone_id,
    get_call_media_type,
    get_call_user,
    register_call_user,
)


class WebRTCSessionRegistryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
