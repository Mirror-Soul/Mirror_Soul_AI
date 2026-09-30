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
[CALL_TRACE] turn completed: callId=31 turn=1 user=user-1 media_type=VIDEO total_ms=1250 stt_ms=100 context_ms=50 rag_ms=80 llm_ms=500 tts_ms=200 video_ms=320
[CALL_TRACE] call closed: callId=31 user=user-1 clone_id=9 media_type=VIDEO status=COMPLETED reason=CALL_END duration_ms=4000 turns_started=1 turns_completed=1 turns_failed=0 turns_skipped=0 turns_cancelled=0 last_error=none
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
        self.assertIn("turn completed", snapshot.trace_summary)
        self.assertEqual(len(snapshot.recent_call_summaries), 1)

    def test_filters_interleaved_events_to_latest_call(self) -> None:
        logs = """
[SIGNALING] received: {'type': 'CALL_INVITE', 'data': {'callId': 40, 'cloneUserUuid': 'old-user', 'mediaType': 'VIDEO'}}
[SIGNALING] CALL_ACCEPT sent: callId=40
[SIGNALING] received: {'type': 'CALL_INVITE', 'data': {'callId': 41, 'cloneUserUuid': 'new-user', 'mediaType': 'VIDEO'}}
[SIGNALING] CALL_ACCEPT sent: callId=41
[REALTIME] STT user=old-user callId=40 turn=2 elapsed_ms=100: 이전 통화
[REALTIME] STT start: callId=41 turn=1 user=new-user wav_bytes=100
[REALTIME] STT user=new-user callId=41 turn=1 elapsed_ms=90: 최신 통화
[CALL_TRACE] turn completed: callId=41 turn=1 user=new-user total_ms=800
[CALL_TRACE] turn completed: callId=40 turn=2 user=old-user total_ms=900
"""

        snapshot = parse_call_logs(logs)

        self.assertEqual(snapshot.call_id, "41")
        self.assertEqual(snapshot.user_uuid, "new-user")
        self.assertEqual(snapshot.stt.detail, "최신 통화")
        self.assertTrue(all("callId=40" not in line for line in snapshot.events))
        self.assertIn("callId=41", snapshot.trace_summary)

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
