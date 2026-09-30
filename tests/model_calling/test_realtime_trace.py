import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from model_calling.realtime.trace import CallTrace, trace_fields


class CallTraceTests(unittest.TestCase):
    def test_counts_each_turn_once_and_prints_call_summary(self) -> None:
        trace = CallTrace(
            call_id=81,
            user_id="member-uuid",
            clone_id=15,
            media_type="VIDEO",
        )
        first = trace.start_turn()
        second = trace.start_turn()
        trace.finish_turn(first, "COMPLETED")
        trace.finish_turn(first, "FAILED", error="duplicate")
        trace.finish_turn(second, "FAILED", error="render failed")

        output = io.StringIO()
        with patch("model_calling.realtime.trace.time.monotonic", return_value=2.5):
            trace.started_at = 1.0
            with redirect_stdout(output):
                trace.close(reason="CALL_END")
                trace.close(reason="DUPLICATE")

        summary = output.getvalue()
        self.assertEqual(summary.count("[CALL_TRACE] call closed:"), 1)
        self.assertIn("callId=81", summary)
        self.assertIn("duration_ms=1500", summary)
        self.assertIn("turns_completed=1", summary)
        self.assertIn("turns_failed=1", summary)
        self.assertIn("last_error=render failed", summary)

    def test_trace_fields_supports_pre_call_operations(self) -> None:
        self.assertEqual(trace_fields(None), "callId=unknown")
        self.assertEqual(trace_fields(10, 3), "callId=10 turn=3")

    def test_failed_connection_reason_marks_call_failed(self) -> None:
        trace = CallTrace(
            call_id=82,
            user_id="member-uuid",
            clone_id=15,
            media_type="VIDEO",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            trace.close(reason="PEER_FAILED")

        self.assertIn("status=FAILED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
