import unittest
from unittest.mock import AsyncMock, Mock, patch

from model_calling.realtime.pipeline import (
    _append_conversation_turn,
    generate_reply_audio,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile


class RealtimeConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_uses_and_updates_bounded_conversation_history(self):
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
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(16)
        ]

        async def process_llm_side_effect(**kwargs):
            self.assertEqual(kwargs["conversation_history"], history)
            return "저도 그 이야기 기억해요."

        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(return_value="아까 이야기 기억나?"),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(return_value=({"name": "회원"}, personality, speech, "ENFP")),
            ),
            patch(
                "model_calling.realtime.pipeline.search_user_memories",
                new=Mock(return_value=[]),
            ),
            patch(
                "model_calling.realtime.pipeline.process_llm",
                new=AsyncMock(side_effect=process_llm_side_effect),
            ),
            patch(
                "model_calling.realtime.pipeline.process_tts_bytes",
                new=AsyncMock(return_value=b"audio"),
            ),
        ):
            result = await generate_reply_audio(
                "member-uuid",
                14,
                b"wav",
                conversation_history=history,
            )

        self.assertEqual(result.audio_bytes, b"audio")
        self.assertEqual(result.user_text, "아까 이야기 기억나?")
        self.assertEqual(result.assistant_text, "저도 그 이야기 기억해요.")
        self.assertEqual(len(history), 16)

        _append_conversation_turn(
            history,
            user_id="member-uuid",
            user_text=result.user_text,
            assistant_text=result.assistant_text,
        )

        self.assertEqual(len(history), 16)
        self.assertEqual(history[-2]["content"], "아까 이야기 기억나?")
        self.assertEqual(history[-1]["content"], "저도 그 이야기 기억해요.")


if __name__ == "__main__":
    unittest.main()
