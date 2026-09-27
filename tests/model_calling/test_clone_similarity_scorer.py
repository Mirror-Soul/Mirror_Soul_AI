import unittest
from datetime import date

from model_calling.clone_similarity.scorer import (
    CloneSimilaritySnapshot,
    calculate_clone_similarity,
)
from shared.clone_similarity import calculate_overall_clone_similarity


def _snapshot(**overrides):
    values = {
        "clone_id": 14,
        "user_uuid": "member-1",
        "name": "회원",
        "gender": "MALE",
        "birth_date": date(1990, 1, 1),
        "job": "개발자",
        "job_description": "서비스를 개발합니다.",
        "self_introduction": "새로운 경험을 좋아합니다.",
        "mbti": "ENFP",
        "voice_profile_id": 7,
        "voice_training_job_id": 11,
        "elevenlabs_voice_id": "voice-1",
        "voice_training_status": "COMPLETED",
        "voice_training_audio_count": 5,
        "interview_answer_count": 5,
        "interview_text_count": 5,
        "interview_audio_count": 5,
        "completed_call_count": 0,
        "user_talk_log_count": 0,
    }
    values.update(overrides)
    return CloneSimilaritySnapshot(**values)


class CloneSimilarityScorerTests(unittest.TestCase):
    def test_complete_onboarding_starts_in_low_sixties_with_face(self) -> None:
        components = calculate_clone_similarity(
            _snapshot(),
            actual_voice_score=62.0,
        )
        result = calculate_overall_clone_similarity(
            face_score=75.0,
            voice_score=components.voice_score,
            profile_score=components.personality_score,
            data_reliability_score=components.data_reliability_score,
            penalty_score=components.penalty_score,
        )

        self.assertEqual(components.personality_score, 51.0)
        self.assertGreaterEqual(result.total_score, 60.0)
        self.assertLess(result.total_score, 65.0)

    def test_more_distinct_usage_raises_personality_component(self) -> None:
        onboarding = calculate_clone_similarity(
            _snapshot(),
            actual_voice_score=62.0,
        )
        mature = calculate_clone_similarity(
            _snapshot(
                voice_training_audio_count=20,
                interview_answer_count=15,
                interview_text_count=15,
                interview_audio_count=15,
                completed_call_count=12,
                user_talk_log_count=40,
            ),
            actual_voice_score=100.0,
        )

        self.assertGreater(mature.personality_score, onboarding.personality_score)
        self.assertEqual(mature.personality_score, 100.0)
        self.assertEqual(mature.data_reliability_score, 100.0)


if __name__ == "__main__":
    unittest.main()
