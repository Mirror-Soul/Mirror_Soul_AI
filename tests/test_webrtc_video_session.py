import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from model_calling.webrtc.peer import create_answer_from_offer
from model_calling.webrtc.session import (
    close_session,
    get_session,
    register_call_user,
)


class _FakePeerConnection:
    def __init__(self) -> None:
        self.tracks = []
        self.localDescription = None
        self.closed = False

    def addTrack(self, track):
        self.tracks.append(track)

    async def setRemoteDescription(self, description):
        self.remote_description = description

    async def createAnswer(self):
        return SimpleNamespace(type="answer", sdp="answer-sdp")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def close(self):
        self.closed = True


class _Renderer:
    def __init__(self) -> None:
        self.name = "renderer"
        self.prepared = False

    async def prepare(self) -> None:
        self.prepared = True


class WebRTCVideoSessionTests(unittest.TestCase):
    def test_video_call_adds_audio_and_video_output_tracks(self) -> None:
        async def run():
            call_id = 7001
            pc = _FakePeerConnection()
            video_renderer = _Renderer()
            register_call_user(call_id, "member-uuid", 6, "VIDEO")
            with patch(
                "model_calling.webrtc.peer.create_peer_connection",
                return_value=pc,
            ), patch(
                "model_calling.webrtc.peer.create_ditto_video_session",
                return_value=video_renderer,
            ):
                answer = await create_answer_from_offer(
                    call_id=call_id,
                    room_id="room",
                    ai_signal_id="ai",
                    caller_signal_id="caller",
                    offer_sdp={"type": "offer", "sdp": "offer-sdp"},
                )
                session = get_session(call_id)
                kinds = [track.kind for track in pc.tracks]
                renderer = session.video_renderer
                prepare_task = session.video_prepare_task
                await prepare_task
            await close_session(call_id)
            return answer, kinds, renderer, pc.closed

        with patch.dict(os.environ, {}, clear=False):
            answer, kinds, renderer, closed = asyncio.run(run())

        self.assertEqual(answer, {"type": "answer", "sdp": "answer-sdp"})
        self.assertEqual(kinds, ["audio", "video"])
        self.assertEqual(renderer.name, "renderer")
        self.assertTrue(renderer.prepared)
        self.assertTrue(closed)

    def test_voice_call_keeps_audio_only(self) -> None:
        async def run():
            call_id = 7002
            pc = _FakePeerConnection()
            register_call_user(call_id, "member-uuid", 6, "VOICE")
            with patch(
                "model_calling.webrtc.peer.create_peer_connection",
                return_value=pc,
            ):
                await create_answer_from_offer(
                    call_id=call_id,
                    room_id="room",
                    ai_signal_id="ai",
                    caller_signal_id="caller",
                    offer_sdp={"type": "offer", "sdp": "offer-sdp"},
                )
                kinds = [track.kind for track in pc.tracks]
            await close_session(call_id)
            return kinds

        self.assertEqual(asyncio.run(run()), ["audio"])


if __name__ == "__main__":
    unittest.main()
