import unittest

from tools.ai_pipeline_monitor import (
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    PipelineSnapshot,
    RemoteResult,
    Stage,
    parse_pipeline_logs,
    render,
)


class AiPipelineMonitorTest(unittest.TestCase):
    def test_parses_all_ai_pipeline_stages_for_member(self) -> None:
        user_uuid = "bb3fe728-136f-43d0-891b-9c4bd82e95be"
        ai_logs = f"""
[RAG_PROFILE] processing: user_uuid={user_uuid} clone_id=13 samples=5
[RAG_PROFILE] completed: user_uuid={user_uuid} clone_id=13 profile_score=64.25 data_reliability=91.5 penalty=1.5 callback_sent=True
[VOICE_TRAINING] processing: job_id=13 user_uuid={user_uuid} clone_id=13 files=5
[VOICE_TRAINING_QUALITY] sample: job_id=13 sample=1 status=ACCEPTED duration=3.20s rms_dbfs=-18.0 silence_ratio=0.12 clipping_ratio=0.0
[VOICE_TRAINING_QUALITY] batch: job_id=13 status=PASSED accepted=5 rejected=0 duration=18.40s
[CLONE_SIMILARITY] updated: user_uuid={user_uuid} clone_id=13 overall=72.4 face=86.5 voice=62.0 personality=64.25 data_reliability=91.5 penalty=1.5 complete=True
[VOICE_TRAINING] completed: job_id=13 clone_id=13 voice_id=Y6tM...YvaS
"""
        gpu_logs = f"""
[FACE_TRAINING] preprocessing: job_id=5 user_uuid={user_uuid} clone_id=13 files=1
[FACE_SIMILARITY] completed: score=86.50 source_preservation=90.00 render=80.00 confidence=high calibrated=True
[FACE_TRAINING] completed: job_id=5 user_uuid={user_uuid} clone_id=13 face_score=86.5 quality_tier=NORMAL profile=s3://bucket/face.json
"""

        snapshot = parse_pipeline_logs(ai_logs, gpu_logs, user_uuid)

        self.assertEqual(snapshot.rag.status, "COMPLETED")
        self.assertEqual(snapshot.rag.score, "64.25")
        self.assertEqual(snapshot.voice.status, "COMPLETED")
        self.assertEqual(snapshot.voice.score, "62.0")
        self.assertTrue(
            any("[VOICE_TRAINING_QUALITY] batch:" in line for line in snapshot.events)
        )
        self.assertEqual(snapshot.face.status, "COMPLETED")
        self.assertEqual(snapshot.face.score, "86.5")
        self.assertEqual(snapshot.overall_score, "67.9")
        self.assertEqual(
            snapshot.overall_note,
            "estimated from complete AI component logs",
        )

    def test_uses_latest_ai_user_when_uuid_is_omitted(self) -> None:
        ai_logs = "\n".join(
            [
                "[VOICE_TRAINING] processing: job_id=1 user_uuid=old clone_id=1 files=1",
                "[RAG_PROFILE] processing: user_uuid=new clone_id=2 samples=3",
            ]
        )

        snapshot = parse_pipeline_logs(ai_logs, "")

        self.assertEqual(snapshot.user_uuid, "new")
        self.assertEqual(snapshot.rag.status, "PROCESSING")

    def test_warns_when_rag_callback_was_not_sent(self) -> None:
        ai_logs = (
            "[RAG_PROFILE] completed: user_uuid=target clone_id=2 "
            "profile_score=60.0 data_reliability=80.0 penalty=0.0 "
            "callback_sent=False"
        )

        snapshot = parse_pipeline_logs(ai_logs, "", "target")

        self.assertEqual(snapshot.rag.status, "WARNING")
        self.assertIn("callback=False", snapshot.rag.detail)

    def test_marks_voice_failed_when_input_quality_batch_fails(self) -> None:
        ai_logs = "\n".join(
            [
                "[VOICE_TRAINING] processing: job_id=21 user_uuid=target clone_id=7 files=5",
                "[VOICE_TRAINING_QUALITY] sample: job_id=21 sample=2 status=REJECTED reasons=too_much_silence",
                "[VOICE_TRAINING_QUALITY] batch: job_id=21 status=FAILED accepted=2 rejected=3 duration=5.20s",
            ]
        )

        snapshot = parse_pipeline_logs(ai_logs, "", "target")

        self.assertEqual(snapshot.voice.status, "FAILED")
        self.assertEqual(snapshot.voice.job_id, "21")
        self.assertIn("accepted=2", snapshot.voice.detail)
        self.assertIn("rejected=3", snapshot.voice.detail)

    def test_scopes_face_events_to_matching_job_block(self) -> None:
        gpu_logs = "\n".join(
            [
                "[FACE_TRAINING] preprocessing: job_id=4 user_uuid=target clone_id=4 files=1",
                "[FACE_TRAINING] completed: job_id=4 user_uuid=target clone_id=4 face_score=81.0 quality_tier=NORMAL profile=s3://first",
                "[FACE_TRAINING] preprocessing: job_id=5 user_uuid=other clone_id=5 files=1",
                "[FACE_TRAINING] job failed: job_id=5 error=render_failed",
            ]
        )

        snapshot = parse_pipeline_logs("", gpu_logs, "target")

        self.assertEqual(snapshot.face.status, "COMPLETED")
        self.assertEqual(snapshot.face.job_id, "4")
        self.assertEqual(snapshot.face.score, "81.0")

    def test_render_colorizes_health_and_pipeline_statuses(self) -> None:
        snapshot = PipelineSnapshot(
            user_uuid="target",
            face=Stage(status="FAILED", detail="render_failed"),
        )

        output = render(
            snapshot,
            RemoteResult(True, ""),
            {"AI_API": "active", "VOICE_WORKER": "active"},
            RemoteResult(True, ""),
            {"FACE_WORKER": "active", "GPU": "100, 1000, 0"},
            color=True,
        )

        self.assertIn(f"{ANSI_GREEN}OK{ANSI_RESET}", output)
        self.assertIn(ANSI_RED, output)
        self.assertIn("FAILED", output)

    def test_render_remains_plain_when_color_is_disabled(self) -> None:
        output = render(
            PipelineSnapshot(user_uuid="target"),
            RemoteResult(True, ""),
            {"AI_API": "active", "VOICE_WORKER": "active"},
            RemoteResult(True, ""),
            {"FACE_WORKER": "active", "GPU": "100, 1000, 0"},
        )

        self.assertNotIn("\033[", output)

    def test_render_marks_cached_server_data_as_stale(self) -> None:
        output = render(
            PipelineSnapshot(user_uuid="target"),
            RemoteResult(False, "", "SSH query timed out"),
            {"AI_API": "active", "VOICE_WORKER": "active"},
            RemoteResult(True, ""),
            {"FACE_WORKER": "active", "GPU": "100, 1000, 0"},
            ai_cached=True,
        )

        self.assertIn("WARNING (SSH query timed out; showing last data)", output)
        self.assertIn("STALE (OK)", output)


if __name__ == "__main__":
    unittest.main()
