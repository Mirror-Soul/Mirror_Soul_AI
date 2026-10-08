import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from model_calling.clients.rag_search import merge_memory_results
from model_calling.realtime.pipeline import (
    _search_memories,
    generate_reply_audio,
    memory_search_queries,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile


PROFILE = {
    "documentId": "u:profile_snapshot:default",
    "text": "[회원 핵심 프로필]",
    "metadata": {"sourceType": "profile_snapshot"},
    "distance": None,
}


def _memory(doc_id, distance, source="interview_memory"):
    return {
        "documentId": doc_id,
        "text": f"text {doc_id}",
        "metadata": {"sourceType": source},
        "distance": distance,
    }


TEAM_HISTORY = [
    {"role": "user", "content": "팀 프로젝트에 비협조적인 팀원이 있으면 어떻게 해?"},
    {"role": "assistant", "content": "빼고 진행할 것 같아."},
]


class MemorySearchQueryTests(unittest.TestCase):
    def test_first_turn_uses_single_query(self):
        self.assertEqual(memory_search_queries("면접 전날 떨리면?", []), ["면접 전날 떨리면?"])

    def test_follow_up_searches_current_question_alone_and_with_context(self):
        queries = memory_search_queries("면접 전날 너무 떨리면 어떻게 해?", TEAM_HISTORY)
        self.assertEqual(queries[0], "면접 전날 너무 떨리면 어떻게 해?")
        self.assertEqual(len(queries), 2)
        self.assertIn("팀 프로젝트", queries[1])
        self.assertNotIn("팀 프로젝트", queries[0])


class MergeMemoryResultsTests(unittest.TestCase):
    def test_profile_once_then_closest_memories(self):
        merged = merge_memory_results(
            [
                [PROFILE, _memory("team", 0.70), _memory("hobby", 0.74)],
                [PROFILE, _memory("interview", 0.45), _memory("team", 0.60)],
            ],
            top_k=2,
        )
        self.assertEqual(
            [m["documentId"] for m in merged],
            ["u:profile_snapshot:default", "interview", "team"],
        )
        team = next(m for m in merged if m["documentId"] == "team")
        self.assertEqual(team["distance"], 0.60)

    def test_memories_without_ids_are_deduplicated_by_text(self):
        a = {"text": "같은  기억", "metadata": {"sourceType": "conversation_memory"}, "distance": 0.5}
        b = {"text": "같은 기억", "metadata": {"sourceType": "conversation_memory"}, "distance": 0.3}
        merged = merge_memory_results([[a], [b]], top_k=5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["distance"], 0.3)


class SearchMemoriesTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_failed_query_keeps_the_other_results(self):
        def search(user_id, query, top_k):
            if query.startswith("현재 질문:"):
                raise RuntimeError("timeout")
            return [PROFILE, _memory("interview", 0.4)]

        with patch("model_calling.realtime.pipeline.search_user_memories", new=search):
            memories, failed = await _search_memories(
                "u", ["면접?", "현재 질문: 면접?\n최근 사용자 발화:\n- 팀"], 6
            )
        self.assertEqual(failed, 1)
        self.assertEqual([m["documentId"] for m in memories][1], "interview")

    async def test_all_queries_failing_raises(self):
        def search(user_id, query, top_k):
            raise RuntimeError("down")

        with patch("model_calling.realtime.pipeline.search_user_memories", new=search):
            with self.assertRaises(RuntimeError):
                await _search_memories("u", ["a", "b"], 6)

    async def test_interview_found_only_by_current_question_reaches_llm(self):
        personality = PersonalityProfile(
            openness=50, conscientiousness=50, extraversion=50,
            agreeableness=50, neuroticism=50, summary="테스트",
        )
        speech = SpeechProfile(
            user_id="u", voice_id="v", speech_speed=50, avg_pitch=50,
            honorific_ratio=50, summary="테스트 말투",
        )
        interview = _memory("u:interview_memory:interview-eve", 0.42)

        def search(user_id, query, top_k):
            if query.startswith("현재 질문:"):
                return [PROFILE, _memory("u:interview_memory:team", 0.6)]
            return [PROFILE, interview]

        llm = AsyncMock(return_value="그냥 포기하고 도망갈 것 같아.")
        with (
            patch("model_calling.realtime.pipeline.process_stt", new=AsyncMock(return_value="면접 전날 너무 떨리면 어떻게 해?")),
            patch("model_calling.realtime.pipeline.load_runtime_context", new=AsyncMock(return_value=({"name": "회원"}, personality, speech, "ISTP"))),
            patch("model_calling.realtime.pipeline.search_user_memories", new=search),
            patch("model_calling.realtime.pipeline.process_llm", new=llm),
            patch("model_calling.realtime.pipeline.process_tts_bytes", new=AsyncMock(return_value=b"audio")),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                await generate_reply_audio(
                    "u", 1, b"wav",
                    conversation_history=list(TEAM_HISTORY),
                    call_id=7, turn_id=3,
                )

        used_ids = [m["documentId"] for m in llm.call_args.kwargs["retrieved_memories"]]
        self.assertIn("u:interview_memory:interview-eve", used_ids)
        self.assertIn("u:interview_memory:team", used_ids)
        log = output.getvalue()
        self.assertIn("queries=2 failed_queries=0", log)
        self.assertIn("sources=interview_memory:2,profile_snapshot:1", log)


if __name__ == "__main__":
    unittest.main()
