import unittest

from tools.ai_pipeline_stream import _ai_command, _gpu_command, format_event


class AiPipelineStreamTest(unittest.TestCase):
    def test_formats_source_and_event_without_color(self) -> None:
        output = format_event(
            "AI",
            "[RAG_PROFILE] completed: user_uuid=user-1 clone_id=1",
            color=False,
        )

        self.assertIn("[AI ]", output)
        self.assertIn("user_uuid=user-1", output)

    def test_stream_commands_follow_only_ai_pipeline_events(self) -> None:
        ai_command = _ai_command(history=20, scan_lines=800)
        gpu_command = _gpu_command("/tmp/face.log", history=20, scan_lines=800)

        self.assertIn("journalctl", ai_command)
        self.assertIn("--line-buffered", ai_command)
        self.assertIn("RAG_PROFILE", ai_command)
        self.assertIn("tail -n 0 -F /tmp/face.log", gpu_command)
        self.assertIn("FACE_(TRAINING|SIMILARITY)", gpu_command)


if __name__ == "__main__":
    unittest.main()
