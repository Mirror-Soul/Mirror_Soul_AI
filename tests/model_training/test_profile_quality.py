import unittest

from model_training.profile_quality import evaluate_profile_quality


def _sample(index: int, transcript: str) -> dict[str, object]:
    return {
        "questionId": index,
        "questionCategory": f"category-{index}",
        "questionText": f"question-{index}",
        "transcript": transcript,
    }


class ProfileQualityTests(unittest.TestCase):
    def test_meaningful_onboarding_builds_conservative_profile_score(self) -> None:
        result = evaluate_profile_quality(
            age=30,
            gender="MALE",
            mbti="ENFP",
            description="새로운 사람과 이야기하는 것을 좋아합니다.",
            interests=["여행", "음악"],
            interview_topics=["관계", "취미"],
            interview_samples=[
                _sample(index, f"제가 중요하게 생각하는 경험에 대한 답변 {index}")
                for index in range(5)
            ],
        )

        self.assertGreaterEqual(result.profile_score, 55.0)
        self.assertLess(result.profile_score, 75.0)
        self.assertGreaterEqual(result.data_reliability_score, 90.0)
        self.assertEqual(result.penalty_score, 0.0)

    def test_repeated_and_placeholder_answers_do_not_raise_quality(self) -> None:
        repeated = "같은 답변을 반복합니다"
        result = evaluate_profile_quality(
            age=30,
            gender="MALE",
            mbti="ENFP",
            description="소개",
            interests=[],
            interview_topics=[],
            interview_samples=[
                _sample(1, repeated),
                _sample(2, repeated),
                _sample(3, "테스트"),
                _sample(4, "asdf"),
            ],
        )

        self.assertEqual(result.meaningful_answer_count, 1)
        self.assertEqual(result.duplicate_answer_count, 1)
        self.assertEqual(result.placeholder_answer_count, 2)
        self.assertEqual(result.penalty_score, 5.5)
        self.assertLess(result.data_reliability_score, 80.0)

    def test_more_distinct_answers_raise_profile_score(self) -> None:
        onboarding = evaluate_profile_quality(
            age=30,
            gender="FEMALE",
            mbti="ISFJ",
            description="가족과 일상을 중요하게 생각합니다.",
            interests=["요리"],
            interview_topics=[],
            interview_samples=[
                _sample(index, f"서로 다른 온보딩 경험을 설명하는 답변 {index}")
                for index in range(5)
            ],
        )
        mature = evaluate_profile_quality(
            age=30,
            gender="FEMALE",
            mbti="ISFJ",
            description="가족과 일상을 중요하게 생각합니다.",
            interests=["요리"],
            interview_topics=[],
            interview_samples=[
                _sample(index, f"서로 다른 경험과 가치관을 구체적으로 설명하는 답변 {index}")
                for index in range(15)
            ],
        )

        self.assertGreater(mature.profile_score, onboarding.profile_score)


if __name__ == "__main__":
    unittest.main()
