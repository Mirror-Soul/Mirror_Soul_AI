import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_training import behavior_profile, services
from model_training.behavior_profile import (
    OTHER_CATEGORY,
    SYSTEM_PROMPT,
    BehaviorTrait,
    build_interview_digest,
    extract_behavior_traits,
    parse_behavior_traits,
)
from model_training.rag_documents import (
    BEHAVIOR_SECTION_HEADING,
    build_profile_snapshot_text,
    replace_behavior_section,
)
from shared.config import settings


SAMPLES = [
    {
        "questionCategory": "협업",
        "questionText": "팀 프로젝트에 비협조적인 팀원이 있으면 어떻게 할 건가요?",
        "transcript": "그냥 그 사람을 내쫓아버리겠습니다.",
    },
    {
        "questionCategory": "스트레스",
        "questionText": "면접 전날 너무 떨리면 어떻게 할 건가요?",
        "transcript": "그냥 포기하고 도망가겠습니다.",
    },
    {"questionText": "빈 답변", "transcript": " "},
]

TRAITS_JSON = json.dumps(
    {
        "traits": [
            {"category": "갈등·협업", "behavior": "비협조적인 팀원은 설득하지 않고 팀에서 내보낸다", "evidence": "그 사람을 내쫓아버리겠습니다"},
            {"category": "압박·스트레스", "behavior": "큰 압박 앞에서는 포기하고 피한다", "evidence": "“포기하고 도망가겠습니다”"},
        ]
    },
    ensure_ascii=False,
)


def _client(content):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return client


class PromptAndDigestTests(unittest.TestCase):
    def test_system_prompt_renders_limit_and_json_example(self):
        prompt = SYSTEM_PROMPT.format(max_traits=5)
        self.assertIn("최대 5개", prompt)
        self.assertIn('{"traits": [{"category": "갈등·협업"', prompt)
        self.assertIn("순화하거나", prompt)

    def test_digest_skips_blank_answers_and_keeps_question(self):
        digest = build_interview_digest(SAMPLES)
        self.assertIn("[1]", digest)
        self.assertIn("[2]", digest)
        self.assertNotIn("[3]", digest)
        self.assertIn("질문: 면접 전날 너무 떨리면 어떻게 할 건가요?", digest)
        self.assertIn("답변: 그냥 포기하고 도망가겠습니다.", digest)


class ParseTests(unittest.TestCase):
    def test_parses_and_normalizes(self):
        traits = parse_behavior_traits(TRAITS_JSON, max_traits=8)
        self.assertEqual(len(traits), 2)
        self.assertEqual(traits[1].evidence, "포기하고 도망가겠습니다")
        self.assertEqual(
            traits[0].to_line(),
            '갈등·협업: 비협조적인 팀원은 설득하지 않고 팀에서 내보낸다 (회원 답변: "그 사람을 내쫓아버리겠습니다")',
        )

    def test_unknown_category_duplicates_and_limit(self):
        content = json.dumps(
            {
                "traits": [
                    {"category": "기타등등", "behavior": "A를 한다"},
                    {"category": "대인관계", "behavior": "A를 한다"},
                    {"category": "대인관계", "behavior": "B를 한다"},
                    {"category": "대인관계", "behavior": "C를 한다"},
                ]
            },
            ensure_ascii=False,
        )
        traits = parse_behavior_traits(content, max_traits=2)
        self.assertEqual([t.behavior for t in traits], ["A를 한다", "B를 한다"])
        self.assertEqual(traits[0].category, OTHER_CATEGORY)

    def test_invalid_json_returns_empty(self):
        self.assertEqual(parse_behavior_traits("not json", 8), [])
        self.assertEqual(parse_behavior_traits('{"traits": "x"}', 8), [])


class ExtractTests(unittest.TestCase):
    def test_calls_llm_in_json_mode_with_answers(self):
        client = _client(TRAITS_JSON)
        traits = extract_behavior_traits(SAMPLES, client=client, model="m", max_traits=4)
        self.assertEqual(len(traits), 2)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "m")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIn("내쫓아버리겠습니다", kwargs["messages"][1]["content"])
        self.assertIn("최대 4개", kwargs["messages"][0]["content"])

    def test_no_answers_makes_no_call(self):
        client = _client(TRAITS_JSON)
        self.assertEqual(extract_behavior_traits([{"transcript": ""}], client=client), [])
        client.chat.completions.create.assert_not_called()


class SnapshotSectionTests(unittest.TestCase):
    def test_section_follows_basic_fields(self):
        text = build_profile_snapshot_text(
            name="동빈",
            mbti="istp",
            description="개발자",
            keywords=["개발"],
            behavior_lines=["갈등·협업: 비협조적인 팀원은 내보낸다"],
        )
        self.assertLess(text.index(BEHAVIOR_SECTION_HEADING), text.index("[핵심 키워드]"))
        self.assertIn("- 갈등·협업: 비협조적인 팀원은 내보낸다", text)

    def test_replace_is_idempotent_and_removable(self):
        base = build_profile_snapshot_text(name="동빈", keywords=["개발"])
        once = replace_behavior_section(base, ["A"])
        twice = replace_behavior_section(once, ["B"])
        self.assertEqual(twice.count(BEHAVIOR_SECTION_HEADING), 1)
        self.assertIn("- B", twice)
        self.assertNotIn("- A", twice)
        self.assertEqual(replace_behavior_section(twice, []), base)
        self.assertEqual(
            replace_behavior_section(base, ["A"]),
            build_profile_snapshot_text(name="동빈", keywords=["개발"], behavior_lines=["A"]),
        )


class TrainingIntegrationTests(unittest.TestCase):
    def test_failure_never_breaks_training(self):
        with (
            patch.object(settings, "PERSONA_BEHAVIOR_PROFILE_ENABLED", True),
            patch.object(services, "extract_behavior_traits", side_effect=RuntimeError("api down")),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                lines, status = services._behavior_profile_lines("u", SAMPLES)
        self.assertEqual((lines, status), ([], "failed"))
        self.assertIn("[RAG_PROFILE] behavior profile skipped: user_uuid=u", output.getvalue())

    def test_disabled_and_no_interviews(self):
        with patch.object(settings, "PERSONA_BEHAVIOR_PROFILE_ENABLED", False):
            self.assertEqual(services._behavior_profile_lines("u", SAMPLES), ([], "disabled"))
        with patch.object(settings, "PERSONA_BEHAVIOR_PROFILE_ENABLED", True):
            self.assertEqual(services._behavior_profile_lines("u", []), ([], "no_interviews"))

    def test_traits_become_lines(self):
        traits = [BehaviorTrait("압박·스트레스", "포기하고 피한다", "도망가겠습니다")]
        with (
            patch.object(settings, "PERSONA_BEHAVIOR_PROFILE_ENABLED", True),
            patch.object(services, "extract_behavior_traits", return_value=traits),
        ):
            lines, status = services._behavior_profile_lines("u", SAMPLES)
        self.assertEqual(status, "ok")
        self.assertEqual(lines, ['압박·스트레스: 포기하고 피한다 (회원 답변: "도망가겠습니다")'])


if __name__ == "__main__":
    unittest.main()
