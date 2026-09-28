import unittest

from tools.realtime_call_monitor import (
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    RemoteResult,
    parse_call_logs,
    render,
)


class RealtimeCallMonitorTest(unittest.TestCase):
    def test_parses_successful_video_reply_pipeline(self) -> None:
        logs = """
[SIGNALING] connected
[SIGNALING] received: {'type': 'CALL_INVITE', 'data': {'callId': 31, 'cloneUserUuid': 'user-1', 'mediaType': 'VIDEO'}}
[SIGNALING] CALL_ACCEPT sent: callId=31
[WEBRTC] peer connection created: callId=31
[WEBRTC] output video track added: callId=31 renderer=enabled
[WEBRTC] session created: callId=31 roomId=room user=user-1 clone_id=9
[WEBRTC] connection: connected
[DITTO_CALL] face profile loaded: user=user-1 clone_id=9 key=face.json
[WEBRTC] idle portrait ready: callId=31
[REALTIME] STT start: user=user-1 wav_bytes=100
[REALTIME] STT user=user-1: 안녕하세요
[REALTIME] RAG lookup complete: user=user-1 count=5
[REALTIME] LLM start: user=user-1
[REALTIME] LLM user=user-1: 반가워요
[REALTIME] TTS start: user=user-1
[REALTIME] TTS complete: user=user-1 audio_bytes=200
[REALTIME] Ditto reply render started: user=user-1
[DITTO_CALL] render completed: bytes=300 seconds=2.5
[REALTIME] Ditto reply video queued: user=user-1
"""

        snapshot = parse_call_logs(logs)

        self.assertEqual(snapshot.signaling_connection, "CONNECTED")
        self.assertEqual(snapshot.call_id, "31")
        self.assertEqual(snapshot.user_uuid, "user-1")
        self.assertEqual(snapshot.media_type, "VIDEO")
        self.assertEqual(snapshot.signal.status, "COMPLETED")
        self.assertEqual(snapshot.webrtc.status, "CONNECTED")
        self.assertEqual(snapshot.stt.status, "COMPLETED")
        self.assertEqual(snapshot.rag.status, "COMPLETED")
        self.assertEqual(snapshot.llm.status, "COMPLETED")
        self.assertEqual(snapshot.tts.status, "COMPLETED")
        self.assertEqual(snapshot.video.status, "COMPLETED")

    def test_marks_video_render_failure(self) -> None:
        logs = """
[SIGNALING] CALL_ACCEPT sent: callId=32
[WEBRTC] Ditto video renderer unavailable: callId=32 error=missing profile
"""

        snapshot = parse_call_logs(logs)

        self.assertEqual(snapshot.video.status, "FAILED")

    def test_render_colorizes_health_and_failure(self) -> None:
        snapshot = parse_call_logs(
            "[SIGNALING] CALL_REJECT sent: callId=33 reason=CLONE_NOT_FOUND"
        )
        metadata = {
            "CALL_SERVICE": "active",
            "TUNNEL_SERVICE": "active",
            "CALL_HEALTH": '{"status":"ok"}',
            "DITTO_READY": (
                '{"status":"ready","engine":{"busy":false,'
                '"gpu":"RTX 4090","renderCount":0,"lastError":null}}'
            ),
        }

        output = render(
            snapshot,
            RemoteResult(True, ""),
            metadata,
            color=True,
        )

        self.assertIn(f"{ANSI_GREEN}OK{ANSI_RESET}", output)
        self.assertIn(ANSI_RED, output)
        self.assertIn("FAILED", output)

    def test_render_marks_cached_call_data_as_stale(self) -> None:
        metadata = {
            "CALL_SERVICE": "active",
            "TUNNEL_SERVICE": "active",
            "CALL_HEALTH": '{"status":"ok"}',
            "DITTO_READY": '{"status":"ready","engine":{"busy":false}}',
        }

        output = render(
            parse_call_logs("[SIGNALING] connected"),
            RemoteResult(False, "", "SSH query timed out"),
            metadata,
            cached=True,
        )

        self.assertIn("WARNING (SSH query timed out; showing last data)", output)
        self.assertIn("STALE (OK)", output)
        self.assertIn("STALE (CONNECTED)", output)


if __name__ == "__main__":
    unittest.main()
