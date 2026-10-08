import unittest

from tools.call_event_text import describe
from tools.monitor_format import event_tag


class TalkLogEventTextTests(unittest.TestCase):
    def test_saved_line(self):
        event = describe(
            "[TALK_LOG] saved: callId=7 turn=2 speaker=USER talkLogId=41 "
            "duplicated=no chars=12"
        )
        self.assertEqual(event.tag, "기록")
        self.assertEqual(event.text, "통화 기록 저장: 사용자 발화 12자 (기록 번호 41)")

    def test_duplicated_line(self):
        event = describe(
            "[TALK_LOG] saved: callId=7 turn=2 speaker=CLONE talkLogId=41 "
            "duplicated=yes chars=12"
        )
        self.assertEqual(event.text, "클론 발화는 이미 저장됨 (중복 요청)")

    def test_failed_line(self):
        event = describe(
            "[TALK_LOG] save failed: callId=7 turn=2 speaker=CLONE "
            "error_code=TALK_LOG_4090 error=rejected"
        )
        self.assertEqual(event.text, "통화 기록 저장 실패: 클론 발화, 오류 TALK_LOG_4090")

    def test_disabled_and_summary_lines(self):
        disabled = describe(
            "[TALK_LOG] saving disabled: callId=7 "
            "reason=AI_INTERNAL_API_KEY is not configured"
        )
        self.assertEqual(
            disabled.text,
            "통화 기록 저장 꺼짐 (AI_INTERNAL_API_KEY is not configured)",
        )
        summary = describe(
            "[TALK_LOG] call summary: callId=7 saved=6 duplicated=0 failed=1 dropped=0"
        )
        self.assertEqual(
            summary.text,
            "통화 기록 정리: 저장 6건, 중복 0건, 실패 1건, 누락 0건",
        )

    def test_conversation_context_is_not_called_history_save(self):
        event = describe(
            "[REALTIME] conversation context updated: callId=7 turn=2 user=u turns=2"
        )
        self.assertEqual(event.text, "대화 맥락 갱신 (지금까지 2턴)")

    def test_monitor_tag(self):
        self.assertEqual(event_tag("[TALK_LOG] saved: callId=7"), "기록")


if __name__ == "__main__":
    unittest.main()
