"""Monitors must show full text (no '…' cuts) and must not stack redraws."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tools import realtime_call_monitor as call_monitor
from tools.ai_pipeline_stream import format_event
from tools.monitor_format import (
    ScreenRefresher,
    display_width,
    tagged_event,
    timing_summary,
    wrap_text,
)
from tools.realtime_call_stream import (
    CallLogFormatter,
    remote_command,
    split_journal_line,
    strip_ansi,
)

ANSWER = (
    "그런 상황이라면 비협조적인 팀원을 제외하고 프로젝트를 진행할 것 같아요. "
    "팀워크가 중요하니까, 협력하지 않는 사람과는 함께할 수 없다고 생각해요. "
    "당신은 어떻게 생각하시나요?"
)
LLM_LINE = (
    "[REALTIME] LLM user=524a1300-11de-4b30-81fd-1c205db30a5f callId=106 "
    f"turn=3 elapsed_ms=1295: {ANSWER}"
)
TRACE_LINE = (
    "[CALL_TRACE] turn completed: callId=106 turn=3 user=u media_type=VIDEO "
    "total_ms=16690 stt_ms=1032 context_ms=0 rag_ms=217 llm_ms=1295 "
    "tts_ms=1587 video_ms=12503"
)


def _joined(text: str) -> str:
    return " ".join(part.strip() for part in strip_ansi(text).splitlines())


class WrapTests(unittest.TestCase):
    def test_korean_is_two_columns(self) -> None:
        self.assertEqual(display_width("가a"), 3)

    def test_wrap_keeps_every_character_within_width(self) -> None:
        lines = wrap_text(ANSWER, 30)
        self.assertEqual(" ".join(lines), ANSWER)
        self.assertTrue(all(display_width(line) <= 30 for line in lines))

    def test_tagged_event_wraps_instead_of_cutting(self) -> None:
        text = tagged_event(LLM_LINE, "", color=False, width=60)
        self.assertNotIn("…:", text)
        self.assertIn(ANSWER.replace(" ", ""), _joined(text).replace(" ", ""))
        for line in text.splitlines():
            self.assertLessEqual(display_width(line), 60)
        continuation = text.splitlines()[1]
        self.assertTrue(continuation.startswith(" " * 9))

    def test_trace_timings_become_seconds(self) -> None:
        self.assertEqual(
            timing_summary(TRACE_LINE),
            "total 16.7s | STT 1.0s | RAG 0.2s | LLM 1.3s | TTS 1.6s | VIDEO 12.5s",
        )
        text = tagged_event(TRACE_LINE, "", color=False)
        self.assertIn("-> total 16.7s", text)
        self.assertNotIn("video_ms=", text)


class DashboardTests(unittest.TestCase):
    def test_call_dashboard_shows_full_answer(self) -> None:
        logs = "\n".join(
            [
                "[SIGNALING] CALL_ACCEPT sent: callId=106 mediaType=VIDEO user=u clone_id=29",
                "[REALTIME] LLM start: callId=106 turn=3 user=u",
                LLM_LINE,
            ]
        )
        snapshot = call_monitor.parse_call_logs(logs)
        with patch.object(call_monitor, "terminal_width", return_value=70), patch(
            "tools.monitor_format.terminal_width", return_value=70
        ):
            frame = call_monitor.render(
                snapshot, call_monitor.RemoteResult(True, ""), {}, color=False
            )
        self.assertIn(ANSWER.replace(" ", ""), frame.replace(" ", "").replace("\n", ""))
        self.assertNotIn("생각하시나요?…", frame)

    def test_refresher_clears_scrollback_and_skips_identical_frames(self) -> None:
        screen = ScreenRefresher(color=True)
        output = io.StringIO()
        with redirect_stdout(output):
            first = screen.show("Title\nUpdated: 10:00:00\nbody")
            same = screen.show("Title\nUpdated: 10:00:05\nbody")
            changed = screen.show("Title\nUpdated: 10:00:10\nnew body")
        self.assertEqual((first, same, changed), (True, False, True))
        self.assertEqual(output.getvalue().count("\033[3J"), 2)


class CallStreamTests(unittest.TestCase):
    def test_journal_timestamp_becomes_local_time(self) -> None:
        with patch.dict(os.environ, {"TZ": "Asia/Seoul"}):
            if hasattr(__import__("time"), "tzset"):
                __import__("time").tzset()
            clock, message = split_journal_line(
                "2026-10-06T08:43:11+0000 ip-10-0-1-180 python[911]: [REALTIME] x"
            )
        self.assertEqual(message, "[REALTIME] x")
        self.assertRegex(clock, r"^\d\d:43:11$")

    def test_transcript_has_call_turn_headers_and_full_speech(self) -> None:
        formatter = CallLogFormatter(color=False, width=80)
        lines = []
        for raw in (
            "[SIGNALING] CALL_ACCEPT sent: callId=106 mediaType=VIDEO user=524a1300-11de-4b30-81fd-1c205db30a5f clone_id=29",
            "[REALTIME] STT user=524a1300-11de-4b30-81fd-1c205db30a5f callId=106 turn=3 elapsed_ms=1032: 너 원래 그런 사람이야?",
            LLM_LINE,
            TRACE_LINE,
            "[CALL_TRACE] call closed: callId=106 user=u status=COMPLETED",
        ):
            lines += formatter.format(raw)
        text = "\n".join(lines)
        self.assertIn("CALL 106", text)
        self.assertIn("call 106 · turn 3", text)
        self.assertIn("[USER  ] 너 원래 그런 사람이야?", text)
        self.assertIn("[CLONE ]", text)
        self.assertIn(ANSWER.replace(" ", ""), text.replace(" ", "").replace("\n", ""))
        self.assertIn("[TURN  ] completed: total 16.7s", text)
        self.assertIn("CALL 106 closed", text)
        self.assertEqual(text.count("call 106 · turn 3"), 1)

    def test_remote_command_drops_frame_noise(self) -> None:
        command = remote_command(100, 2000)
        self.assertIn("-o short-iso", command)
        self.assertIn("grep --line-buffered -v 'sending video frame'", command)
        self.assertIn("Traceback", command)

    def test_ai_stream_event_is_not_cut(self) -> None:
        line = "[FACE_TRAINING] completed: job_id=16 user_uuid=1e7b3b01-1a84-4cd4-a1ea-4792f1143d55 clone_id=30 face_score=85.21 quality_tier=NORMAL profile=s3://bucket/face-results/x/job-16/face-profile.json"
        text = format_event("GPU", line, color=False, width=70)
        self.assertIn("face-profile.json", _joined(text))
        self.assertNotIn("face-pro…", text)


if __name__ == "__main__":
    unittest.main()
