import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from model_calling.realtime.pipeline import (
    RealtimePipelineError,
    _is_utterance_stale,
    _put_latest_utterance,
    _run_stage,
    QueuedUtterance,
    generate_reply_audio,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile


def _runtime_context():
    personality = PersonalityProfile(
        openness=50,
        conscientiousness=50,
        extraversion=50,
        agreeableness=50,
        neuroticism=50,
        summary="테스트 성격",
    )
    speech = SpeechProfile(
        user_id="member-uuid",
        voice_id="voice-id",
        speech_speed=50,
        avg_pitch=50,
        honorific_ratio=50,
        summary="테스트 말투",
    )
    return {"name": "회원"}, personality, speech, "ENFP"


class RealtimeGuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_timeout_has_stable_error_code(self) -> None:
        async def slow_operation():
            await asyncio.sleep(1)

        with self.assertRaises(RealtimePipelineError) as raised:
            await _run_stage("STT", 0.001, slow_operation)

        self.assertEqual(raised.exception.stage, "STT")
        self.assertEqual(raised.exception.code, "STT_TIMEOUT")

    async def test_empty_llm_response_is_rejected(self) -> None:
        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(return_value="안녕하세요"),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(return_value=_runtime_context()),
            ),
            patch(
                "model_calling.realtime.pipeline.search_user_memories",
                new=Mock(return_value=[]),
            ),
            patch(
                "model_calling.realtime.pipeline.process_llm",
                new=AsyncMock(return_value="   "),
            ),
        ):
            with self.assertRaises(RealtimePipelineError) as raised:
                await generate_reply_audio("member-uuid", 1, b"wav")

        self.assertEqual(raised.exception.code, "LLM_EMPTY_RESPONSE")

    async def test_empty_tts_audio_is_rejected(self) -> None:
        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(return_value="안녕하세요"),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(return_value=_runtime_context()),
            ),
            patch(
                "model_calling.realtime.pipeline.search_user_memories",
                new=Mock(return_value=[]),
            ),
            patch(
                "model_calling.realtime.pipeline.process_llm",
                new=AsyncMock(return_value="반가워요"),
            ),
            patch(
                "model_calling.realtime.pipeline.process_tts_bytes",
                new=AsyncMock(return_value=b""),
            ),
        ):
            with self.assertRaises(RealtimePipelineError) as raised:
                await generate_reply_audio("member-uuid", 1, b"wav")

        self.assertEqual(raised.exception.code, "TTS_EMPTY_AUDIO")

    def test_stale_utterance_boundary(self) -> None:
        self.assertFalse(_is_utterance_stale(20_000, 20.0))
        self.assertTrue(_is_utterance_stale(20_001, 20.0))
        self.assertFalse(_is_utterance_stale(999_999, 0.0))

    def test_full_queue_replaces_oldest_utterance(self) -> None:
        queue = asyncio.Queue(maxsize=1)
        old = QueuedUtterance(b"old", 1.0)
        latest = QueuedUtterance(b"latest", 2.0)
        queue.put_nowait(old)

        replaced = _put_latest_utterance(queue, latest)

        self.assertTrue(replaced)
        self.assertIs(queue.get_nowait(), latest)


if __name__ == "__main__":
    unittest.main()
