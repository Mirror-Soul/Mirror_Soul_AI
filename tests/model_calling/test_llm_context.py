import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from model_calling.schemas import PersonalityProfile, SpeechProfile
from model_calling.services import process_llm
from shared.config import settings


class LlmContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_llm_sends_only_the_configured_recent_turns(self):
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
            speech_speed=50,
            avg_pitch=50,
            honorific_ratio=50,
            summary="테스트 말투",
        )
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
            for index in range(20)
        ]
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="응답"))]
            )
        )

        with (
            patch("model_calling.services.client.chat.completions.create", new=create),
            patch.object(settings, "LLM_MODEL", "gpt-4o-mini"),
            patch.object(settings, "LLM_TEMPERATURE", 0.4),
            patch.object(settings, "LLM_MAX_OUTPUT_TOKENS", 200),
            patch.object(settings, "REALTIME_HISTORY_MAX_TURNS", 8),
        ):
            response = await process_llm(
                user_text="현재 질문",
                user_persona={"name": "회원"},
                personality=personality,
                speech=speech,
                conversation_history=history,
            )

        self.assertEqual(response, "응답")
        messages = create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"], "message-4")
        self.assertEqual(messages[-2]["content"], "message-19")
        self.assertEqual(messages[-1], {"role": "user", "content": "현재 질문"})
        self.assertEqual(create.await_args.kwargs["model"], "gpt-4o-mini")
        self.assertEqual(create.await_args.kwargs["temperature"], 0.4)
        self.assertEqual(create.await_args.kwargs["max_tokens"], 200)

    async def test_reasoning_model_uses_compatible_completion_parameters(self):
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
            speech_speed=50,
            avg_pitch=50,
            honorific_ratio=50,
            summary="테스트 말투",
        )
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="응답"))]
            )
        )

        with (
            patch("model_calling.services.client.chat.completions.create", new=create),
            patch.object(settings, "LLM_MODEL", "gpt-6-luna"),
            patch.object(settings, "LLM_REASONING_EFFORT", "none"),
            patch.object(settings, "LLM_MAX_OUTPUT_TOKENS", 200),
        ):
            await process_llm(
                user_text="현재 질문",
                user_persona={"name": "회원"},
                personality=personality,
                speech=speech,
            )

        request = create.await_args.kwargs
        self.assertEqual(request["model"], "gpt-6-luna")
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["max_completion_tokens"], 200)
        self.assertNotIn("temperature", request)
        self.assertNotIn("max_tokens", request)


if __name__ == "__main__":
    unittest.main()


class BehaviorProfilePromptTests(unittest.TestCase):
    def test_prompt_tells_model_not_to_soften_member_attitudes(self):
        from model_calling.services import build_dynamic_persona_prompt
        from model_calling.schemas import PersonalityProfile, SpeechProfile

        prompt = build_dynamic_persona_prompt(
            {"name": "동빈"},
            PersonalityProfile(openness=50, conscientiousness=50, extraversion=50, agreeableness=50, neuroticism=50, summary="x"),
            SpeechProfile(user_id="u", voice_id="v", speech_speed=50, avg_pitch=50, honorific_ratio=50, summary="기본 말투"),
            retrieved_memories=[],
        )
        self.assertIn("[성향 재현 원칙]", prompt)
        self.assertIn("미화하거나", prompt)
        self.assertIn("모범 답안", prompt)
        self.assertIn("위협", prompt)

    def test_long_profile_keeps_behavior_section(self):
        from model_calling.services import format_retrieved_memories

        profile_text = "[회원 핵심 프로필] " + "가" * 900 + " [인터뷰로 확인된 행동 성향] - 압박·스트레스: 포기하고 피한다"
        text = format_retrieved_memories(
            [{"text": profile_text, "metadata": {"sourceType": "profile_snapshot"}}],
            max_chars=3000,
        )
        self.assertIn("포기하고 피한다", text)
