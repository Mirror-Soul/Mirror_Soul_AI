"""Monitors follow the current server log format (result queue, call context API)."""

import unittest

from tools.ai_pipeline_monitor import RemoteResult, parse_pipeline_logs, render
from tools.monitor_format import compact_event, event_tag, tagged_event
from tools import realtime_call_monitor as call_monitor

USER = "9fbb1dd7-408c-47ac-bd67-c1efb165ee66"
CALL_USER = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"


class AiPipelineMonitorRefreshTests(unittest.TestCase):
    def test_voice_score_comes_from_completed_line(self) -> None:
        ai = f"""
[VOICE_TRAINING] processing: job_id=21 user_uuid={USER} files=5
[VOICE_TRAINING] status published: job_id=21 status=PROCESSING message_id=a
[VOICE_TRAINING] completed: job_id=21 voice_id=Y6tM...YvaS voice_score=71.0
[VOICE_TRAINING] status published: job_id=21 status=COMPLETED message_id=b
"""
        snapshot = parse_pipeline_logs(ai, "", USER)
        self.assertEqual(snapshot.voice.status, "COMPLETED")
        self.assertEqual(snapshot.voice.score, "71.0")
        self.assertIn("voice_score=71.0", snapshot.voice.detail)

    def test_overall_estimate_without_clone_similarity_event(self) -> None:
        ai = f"""
[RAG_PROFILE] completed: user_uuid={USER} clone_id=18 profile_score=64.25 data_reliability=91.5 penalty=1.5 documents=6 removed=0 callback_sent=True
[VOICE_TRAINING] processing: job_id=21 user_uuid={USER} files=5
[VOICE_TRAINING] completed: job_id=21 voice_id=x voice_score=71.0
"""
        gpu = f"""
[FACE_TRAINING] preprocessing: job_id=9 user_uuid={USER} clone_id=18 files=1
[FACE_TRAINING] completed: job_id=9 user_uuid={USER} clone_id=18 face_score=83.33 quality_tier=NORMAL profile=s3://b/face.json
"""
        snapshot = parse_pipeline_logs(ai, gpu, USER)
        self.assertNotEqual(snapshot.overall_score, "-")
        self.assertIn("docs=6", snapshot.rag.detail)

    def test_result_queue_failure_is_failed(self) -> None:
        ai = f"""
[VOICE_TRAINING] processing: job_id=22 user_uuid={USER} files=5
[VOICE_TRAINING] status published: job_id=22 status=FAILED message_id=c
"""
        snapshot = parse_pipeline_logs(ai, "", USER)
        self.assertEqual(snapshot.voice.status, "FAILED")

    def test_restarting_worker_is_warning(self) -> None:
        snapshot = parse_pipeline_logs("", "", USER)
        output = render(
            snapshot,
            RemoteResult(True, ""),
            {"AI_API": "active", "VOICE_WORKER": "activating"},
            RemoteResult(True, ""),
            {"FACE_WORKER": "active"},
        )
        self.assertIn("Voice worker: WARNING (restarting", output)


class RealtimeCallMonitorRefreshTests(unittest.TestCase):
    def test_media_type_and_user_from_accept_log(self) -> None:
        logs = f"""
[SIGNALING] received: {{'type': 'CALL_INVITE', 'roomId': 'r', 'data': {{'callId': 91}}}}
[SIGNALING] CALL_ACCEPT sent: callId=91 mediaType=VOICE user={CALL_USER} clone_id=6
[WEBRTC] connection: callId=91 state=connected
"""
        snapshot = call_monitor.parse_call_logs(logs)
        self.assertEqual(snapshot.call_id, "91")
        self.assertEqual(snapshot.media_type, "VOICE")
        self.assertEqual(snapshot.user_uuid, CALL_USER)
        self.assertEqual(snapshot.video.status, "SKIPPED")
        self.assertEqual(snapshot.signal.status, "COMPLETED")

    def test_media_type_from_trace_snake_case(self) -> None:
        logs = """
[SIGNALING] CALL_ACCEPT sent: callId=93
[CALL_TRACE] turn completed: callId=93 turn=1 user=u media_type=VIDEO total_ms=10
"""
        self.assertEqual(call_monitor.parse_call_logs(logs).media_type, "VIDEO")

    def test_rejected_call_shows_call_id_and_reason(self) -> None:
        logs = """
[SIGNALING] received: {'type': 'CALL_INVITE', 'roomId': 'r', 'data': {'callId': 92}}
[SIGNALING] CALL_REJECT sent: callId=92 reason=CLONE_NOT_READY
"""
        snapshot = call_monitor.parse_call_logs(logs)
        self.assertEqual(snapshot.call_id, "92")
        self.assertEqual(snapshot.signal.status, "FAILED")
        self.assertIn("CLONE_NOT_READY", snapshot.signal.detail)


class RagSearchMonitorTests(unittest.TestCase):
    def test_empty_rag_result_is_warning(self) -> None:
        logs = """
[SIGNALING] CALL_ACCEPT sent: callId=95 mediaType=VOICE user=u clone_id=1
[REALTIME] RAG lookup complete: callId=95 turn=1 user=u mode=local count=0 best_distance=none sources=none
"""
        snapshot = call_monitor.parse_call_logs(logs)
        self.assertEqual(snapshot.rag.status, "WARNING")
        self.assertIn("count=0", snapshot.rag.detail)

    def test_rag_search_mode_from_health(self) -> None:
        def output(mode: str) -> str:
            return call_monitor.render(
                call_monitor.parse_call_logs(""),
                call_monitor.RemoteResult(True, ""),
                {"CALL_SERVICE": "active", "CALL_HEALTH": f'{{"status":"ok","ragSearch":"{mode}"}}'},
            )

        self.assertIn("RAG search   : OK (AI server store)", output("remote"))
        self.assertIn("RAG search   : WARNING (local store", output("local"))


class MonitorFormatTests(unittest.TestCase):
    def test_event_tags(self) -> None:
        self.assertEqual(event_tag("[RAG_PROFILE] completed: x"), "RAG")
        self.assertEqual(event_tag("[VOICE_TRAINING_QUALITY] batch: x"), "VOICE")
        self.assertEqual(event_tag("[FACE_SIMILARITY] completed: x"), "FACE")
        self.assertEqual(event_tag("[REALTIME] STT start: x"), "STT")
        self.assertEqual(event_tag("[REALTIME] RAG lookup complete: x"), "RAG")
        self.assertEqual(event_tag("[REALTIME] Ditto reply render started"), "VIDEO")
        self.assertEqual(event_tag("[CALL_TRACE] turn completed: x"), "TRACE")

    def test_compact_event_shortens_uuid_path_and_width(self) -> None:
        line = (
            f"[FACE_TRAINING] completed: user_uuid={USER} "
            f"profile=s3://bucket/face-results/{USER}/job-9/face-profile.json"
        )
        text = compact_event(line)
        self.assertNotIn("[FACE_TRAINING]", text)
        self.assertIn("user_uuid=9fbb1dd7…", text)
        self.assertIn("s3://…/job-9/face-profile.json", text)
        self.assertLessEqual(len(compact_event(line, 40)), 40)

    def test_tagged_event_plain_and_colored(self) -> None:
        plain = tagged_event("[RAG_PROFILE] completed: a=1", "", color=False)
        self.assertTrue(plain.startswith("[RAG   ] 성격·기억(RAG) 학습 완료"))
        raw = tagged_event("[RAG_PROFILE] completed: a=1", "", color=False, korean=False)
        self.assertTrue(raw.startswith("[RAG   ] completed"))
        colored = tagged_event("[RAG_PROFILE] completed: a=1", "\033[32m", color=True)
        self.assertIn("\033[", colored)


if __name__ == "__main__":
    unittest.main()
