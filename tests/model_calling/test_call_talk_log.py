import asyncio
import io
import unittest
import wave
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from model_calling.clients.backend_talk_log import (
    TalkLogSaveError,
    TalkLogSaveResult,
)
from model_calling.realtime.talk_log import (
    CallTalkLogRecorder,
    spoken_window,
    wav_duration_seconds,
)


NOW = datetime(2026, 10, 8, 3, 0, 0, tzinfo=timezone.utc)


def _wav(seconds: float, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeSaver:
    def __init__(self, *, fail_speakers=(), delay=0.0):
        self.calls = []
        self.fail_speakers = set(fail_speakers)
        self.delay = delay

    async def __call__(self, call_id, entry):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append((call_id, entry))
        if entry.speaker in self.fail_speakers:
            raise TalkLogSaveError("rejected", code="TALK_LOG_4090")
        return TalkLogSaveResult(talk_log_id=len(self.calls), duplicated=False)


def _record_turn(recorder, turn_id=1, user="안녕?", clone="안녕하세요!"):
    recorder.record_turn(
        turn_id=turn_id,
        user_text=user,
        user_started_at=NOW,
        user_ended_at=NOW + timedelta(seconds=1),
        clone_text=clone,
        clone_started_at=NOW + timedelta(seconds=3),
        clone_ended_at=NOW + timedelta(seconds=5),
    )


class WavTimingTests(unittest.TestCase):
    def test_wav_duration_reads_header(self):
        self.assertAlmostEqual(wav_duration_seconds(_wav(1.5)), 1.5, places=3)
        self.assertAlmostEqual(wav_duration_seconds(_wav(2.0, rate=48_000)), 2.0, places=3)

    def test_wav_duration_falls_back_for_raw_bytes(self):
        self.assertAlmostEqual(wav_duration_seconds(b"\x00" * (44 + 32_000)), 1.0)

    def test_spoken_window_ends_at_capture_time(self):
        started, ended = spoken_window(ended_at=NOW, duration_seconds=2.0)
        self.assertEqual(ended, NOW)
        self.assertEqual(started, NOW - timedelta(seconds=2))


class CallTalkLogRecorderTests(unittest.TestCase):
    def test_turn_is_saved_user_first_then_clone(self):
        async def run():
            saver = FakeSaver()
            recorder = CallTalkLogRecorder(77, saver=saver, enabled=True)
            _record_turn(recorder)
            await recorder.close(timeout_seconds=1)
            return saver, recorder

        output = io.StringIO()
        with redirect_stdout(output):
            saver, recorder = asyncio.run(run())

        self.assertEqual([entry.speaker for _, entry in saver.calls], ["USER", "CLONE"])
        self.assertEqual({call_id for call_id, _ in saver.calls}, {77})
        self.assertEqual(saver.calls[0][1].message, "안녕?")
        self.assertEqual(saver.calls[1][1].started_at, NOW + timedelta(seconds=3))
        self.assertEqual(recorder.saved, 2)
        log = output.getvalue()
        self.assertIn("[TALK_LOG] saved: callId=77 turn=1 speaker=USER", log)
        self.assertIn("saved=2 duplicated=0 failed=0 dropped=0", log)
        # The conversation text itself is not repeated in the talk log lines.
        self.assertNotIn("안녕하세요", log)

    def test_each_entry_gets_its_own_event_id(self):
        async def run():
            saver = FakeSaver()
            recorder = CallTalkLogRecorder(1, saver=saver, enabled=True)
            _record_turn(recorder, turn_id=1)
            _record_turn(recorder, turn_id=2)
            await recorder.close(timeout_seconds=1)
            return saver

        with redirect_stdout(io.StringIO()):
            saver = asyncio.run(run())
        event_ids = [entry.event_id for _, entry in saver.calls]
        self.assertEqual(len(event_ids), 4)
        self.assertEqual(len(set(event_ids)), 4)

    def test_failure_does_not_stop_following_entries(self):
        async def run():
            saver = FakeSaver(fail_speakers={"USER"})
            recorder = CallTalkLogRecorder(5, saver=saver, enabled=True)
            _record_turn(recorder)
            await recorder.close(timeout_seconds=1)
            return saver, recorder

        output = io.StringIO()
        with redirect_stdout(output):
            saver, recorder = asyncio.run(run())
        self.assertEqual(len(saver.calls), 2)
        self.assertEqual((recorder.saved, recorder.failed), (1, 1))
        self.assertIn("save failed: callId=5 turn=1 speaker=USER error_code=TALK_LOG_4090", output.getvalue())

    def test_recording_does_not_wait_for_backend(self):
        async def run():
            saver = FakeSaver(delay=0.2)
            recorder = CallTalkLogRecorder(5, saver=saver, enabled=True)
            loop = asyncio.get_running_loop()
            started = loop.time()
            _record_turn(recorder)
            elapsed = loop.time() - started
            await recorder.close(timeout_seconds=2)
            return elapsed, saver

        with redirect_stdout(io.StringIO()):
            elapsed, saver = asyncio.run(run())
        self.assertLess(elapsed, 0.05)
        self.assertEqual(len(saver.calls), 2)

    def test_close_timeout_drops_remaining_entries(self):
        async def run():
            saver = FakeSaver(delay=0.5)
            recorder = CallTalkLogRecorder(9, saver=saver, enabled=True)
            _record_turn(recorder)
            await recorder.close(timeout_seconds=0.05)
            return saver, recorder

        output = io.StringIO()
        with redirect_stdout(output):
            saver, recorder = asyncio.run(run())
        self.assertEqual(recorder.dropped, 2)
        self.assertEqual(recorder.pending, 0)
        self.assertIn("dropped=2", output.getvalue())

    def test_entries_after_close_are_dropped(self):
        async def run():
            saver = FakeSaver()
            recorder = CallTalkLogRecorder(9, saver=saver, enabled=True)
            await recorder.close(timeout_seconds=1)
            _record_turn(recorder)
            await asyncio.sleep(0)
            return saver, recorder

        with redirect_stdout(io.StringIO()):
            saver, recorder = asyncio.run(run())
        self.assertEqual(saver.calls, [])
        self.assertEqual(recorder.dropped, 2)

    def test_blank_messages_are_skipped(self):
        async def run():
            saver = FakeSaver()
            recorder = CallTalkLogRecorder(9, saver=saver, enabled=True)
            _record_turn(recorder, user="   ", clone="네")
            await recorder.close(timeout_seconds=1)
            return saver

        with redirect_stdout(io.StringIO()):
            saver = asyncio.run(run())
        self.assertEqual([entry.speaker for _, entry in saver.calls], ["CLONE"])

    def test_disabled_recorder_sends_nothing_and_logs_once(self):
        async def run():
            saver = FakeSaver()
            recorder = CallTalkLogRecorder(3, saver=saver, enabled=False)
            _record_turn(recorder)
            _record_turn(recorder, turn_id=2)
            await recorder.close(timeout_seconds=1)
            return saver

        output = io.StringIO()
        with redirect_stdout(output):
            saver = asyncio.run(run())
        self.assertEqual(saver.calls, [])
        self.assertEqual(output.getvalue().count("saving disabled"), 1)
        self.assertNotIn("call summary", output.getvalue())


class PipelineTalkLogTurnTests(unittest.TestCase):
    def test_reply_timing_follows_audio_queue(self):
        from model_calling.realtime.pipeline import GeneratedReply, _record_talk_log_turn

        recorded = {}

        class Recorder:
            call_id = 7

            def record_turn(self, **kwargs):
                recorded.update(kwargs)

        reply = GeneratedReply(
            audio_bytes=b"mp3",
            user_text="오늘 뭐 했어?",
            assistant_text="산책했어요.",
            started_at=0.0,
            stage_timings_ms={},
        )
        _record_talk_log_turn(
            Recorder(),
            turn_id=4,
            reply=reply,
            user_started_at=NOW - timedelta(seconds=2),
            user_ended_at=NOW,
            reply_queued_at=NOW + timedelta(seconds=3),
            audio_timing=SimpleNamespace(start_delay_seconds=1.5, duration_seconds=2.0),
        )
        self.assertEqual(recorded["turn_id"], 4)
        self.assertEqual(recorded["user_text"], "오늘 뭐 했어?")
        self.assertEqual(recorded["user_started_at"], NOW - timedelta(seconds=2))
        self.assertEqual(recorded["clone_text"], "산책했어요.")
        self.assertEqual(recorded["clone_started_at"], NOW + timedelta(seconds=4.5))
        self.assertEqual(recorded["clone_ended_at"], NOW + timedelta(seconds=6.5))

    def test_unknown_timing_falls_back_to_queue_time(self):
        from unittest.mock import Mock

        from model_calling.realtime.pipeline import GeneratedReply, _record_talk_log_turn

        recorded = {}

        class Recorder:
            call_id = 7

            def record_turn(self, **kwargs):
                recorded.update(kwargs)

        reply = GeneratedReply(b"a", "질문", "답변", 0.0, {})
        _record_talk_log_turn(
            Recorder(),
            turn_id=1,
            reply=reply,
            user_started_at=None,
            user_ended_at=None,
            reply_queued_at=NOW,
            audio_timing=Mock(),
        )
        self.assertEqual(recorded["user_started_at"], NOW)
        self.assertEqual(recorded["user_ended_at"], NOW)
        self.assertEqual(recorded["clone_started_at"], NOW)
        self.assertIsNone(recorded["clone_ended_at"])


if __name__ == "__main__":
    unittest.main()
