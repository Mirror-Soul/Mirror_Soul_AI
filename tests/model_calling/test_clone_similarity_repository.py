import sys
import unittest
from types import SimpleNamespace

try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

from model_calling.clone_similarity.scorer import CloneSimilarityScore
from model_calling.repository.clone_similarity_repository import (
    _save_component_scores,
)


class _ComponentCursor:
    def __init__(
        self,
        *,
        face_score: float | None = None,
        profile_score: float | None = None,
        data_reliability_score: float | None = None,
        penalty_score: float | None = None,
        missing_columns: bool = False,
    ) -> None:
        self.face_score = face_score
        self.profile_score = profile_score
        self.data_reliability_score = data_reliability_score
        self.penalty_score = penalty_score
        self.missing_columns = missing_columns
        self.calls = []

    def execute(self, query, params) -> None:
        self.calls.append((" ".join(query.split()), params))
        if "face_similarity_score" in query and self.missing_columns:
            raise Exception(1054, "Unknown column 'face_similarity_score'")

    def fetchone(self):
        return (
            self.face_score,
            self.profile_score,
            self.data_reliability_score,
            self.penalty_score,
        )


def _score(total_score: float = 72.0) -> CloneSimilarityScore:
    return CloneSimilarityScore(
        clone_id=14,
        voice_profile_id=7,
        voice_training_job_id=11,
        total_score=total_score,
        voice_score=88.0,
        interview_score=62.0,
        profile_score=70.0,
        personality_score=80.0,
        data_reliability_score=90.0,
        penalty_score=0.0,
        explanation="test",
    )


class CloneSimilarityRepositoryTests(unittest.TestCase):
    def test_saves_voice_component_and_combined_sync_rate(self) -> None:
        cursor = _ComponentCursor(face_score=84.0)

        aggregate, columns_available = _save_component_scores(
            cursor,
            _score(72.0),
        )

        self.assertTrue(columns_available)
        self.assertEqual(aggregate.raw_score, 84.6)
        self.assertEqual(aggregate.total_score, 80.4)
        self.assertTrue(aggregate.complete)
        update_query, update_params = cursor.calls[-1]
        self.assertIn("SET voice_similarity_score = %s", update_query)
        self.assertEqual(update_params, (88.0, 80.0, 90.0, 0.0, 80.4, 14))

    def test_falls_back_to_legacy_sync_rate_without_new_columns(self) -> None:
        cursor = _ComponentCursor(missing_columns=True)

        aggregate, columns_available = _save_component_scores(
            cursor,
            _score(63.25),
        )

        self.assertFalse(columns_available)
        self.assertEqual(aggregate.total_score, 56.4)
        fallback_query, fallback_params = cursor.calls[-1]
        self.assertIn("SET sync_rate = %s", fallback_query)
        self.assertNotIn("voice_similarity_score", fallback_query)
        self.assertEqual(fallback_params, (63.2, 14))


if __name__ == "__main__":
    unittest.main()
