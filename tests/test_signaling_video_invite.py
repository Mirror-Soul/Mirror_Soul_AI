import asyncio
import json
import unittest
from unittest.mock import patch

from model_calling.repository.clone_repository import CloneInfo
from model_calling.signaling.handlers import handle_call_invite
from model_calling.webrtc.session import (
    close_session,
    get_call_media_type,
)


class _WebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send(self, message: str) -> None:
        self.messages.append(json.loads(message))


class SignalingVideoInviteTests(unittest.TestCase):
    def test_video_invite_registers_video_media_type(self) -> None:
        async def run():
            ws = _WebSocket()
            message = {
                "type": "CALL_INVITE",
                "roomId": "room",
                "from": "caller",
                "to": "ai",
                "data": {
                    "callId": 8001,
                    "cloneUserUuid": "member-uuid",
                    "mediaType": "video",
                },
            }
            clone = CloneInfo(
                clone_id=6,
                clone_user_uuid="member-uuid",
                sync_rate=80,
                avatar_image_url=None,
                summary=None,
            )
            with patch(
                "model_calling.signaling.handlers.find_clone_by_user_uuid",
                return_value=clone,
            ):
                await handle_call_invite(ws, message)
            media_type = get_call_media_type(8001)
            await close_session(8001)
            return ws.messages, media_type

        messages, media_type = asyncio.run(run())

        self.assertEqual(messages[0]["type"], "CALL_ACCEPT")
        self.assertEqual(messages[0]["data"]["mediaType"], "VIDEO")
        self.assertEqual(media_type, "VIDEO")

    def test_invalid_media_type_is_rejected_before_clone_lookup(self) -> None:
        async def run():
            ws = _WebSocket()
            message = {
                "type": "CALL_INVITE",
                "roomId": "room",
                "from": "caller",
                "to": "ai",
                "data": {
                    "callId": 8002,
                    "cloneUserUuid": "member-uuid",
                    "mediaType": "SCREEN",
                },
            }
            with patch(
                "model_calling.signaling.handlers.find_clone_by_user_uuid"
            ) as lookup:
                await handle_call_invite(ws, message)
            return ws.messages, lookup.call_count

        messages, lookup_calls = asyncio.run(run())

        self.assertEqual(messages[0]["type"], "CALL_REJECT")
        self.assertEqual(messages[0]["data"]["reason"], "INVALID_MEDIA_TYPE")
        self.assertEqual(lookup_calls, 0)


if __name__ == "__main__":
    unittest.main()
