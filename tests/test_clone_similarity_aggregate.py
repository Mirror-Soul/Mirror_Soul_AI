import unittest

from shared.clone_similarity import (
    CLONE_SIMILARITY_CALCULATION_VERSION,
    calculate_overall_clone_similarity,
    face_similarity_component_contract,
)


class OverallCloneSimilarityTests(unittest.TestCase):
    def test_combines_four_components_and_scales_display_score(self) -> None:
        result = calculate_overall_clone_similarity(
            face_score=84.0,
            voice_score=72.0,
            profile_score=80.0,
            data_reliability_score=90.0,
        )

        self.assertEqual(result.raw_score, 79.8)
        self.assertEqual(result.total_score, 75.8)
        self.assertTrue(result.complete)
        self.assertEqual(
            result.to_dict()["calculationVersion"],
            CLONE_SIMILARITY_CALCULATION_VERSION,
        )

    def test_missing_components_are_not_reweighted(self) -> None:
        result = calculate_overall_clone_similarity(
            face_score=None,
            voice_score=63.25,
        )

        self.assertEqual(result.raw_score, 18.97)
        self.assertEqual(result.total_score, 18.0)
        self.assertFalse(result.complete)

    def test_penalty_is_applied_before_display_scaling(self) -> None:
        result = calculate_overall_clone_similarity(
            face_score=80.0,
            voice_score=80.0,
            profile_score=80.0,
            data_reliability_score=80.0,
            penalty_score=5.0,
        )

        self.assertEqual(result.raw_score, 75.0)
        self.assertEqual(result.total_score, 71.3)

    def test_perfect_components_never_exceed_ninety_five(self) -> None:
        result = calculate_overall_clone_similarity(
            face_score=100.0,
            voice_score=100.0,
            profile_score=100.0,
            data_reliability_score=100.0,
        )

        self.assertEqual(result.total_score, 95.0)

    def test_requires_at_least_one_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            calculate_overall_clone_similarity(
                face_score=None,
                voice_score=None,
            )

    def test_face_event_contract_carries_policy_and_weights(self) -> None:
        contract = face_similarity_component_contract(
            {
                "score": 82.5,
                "confidence": "high",
                "calibrationVersion": "member-v1",
                "calibrated": True,
            }
        )

        self.assertEqual(contract["faceScore"], 82.5)
        self.assertEqual(
            contract["weights"],
            {
                "face": 0.3,
                "voice": 0.3,
                "profile": 0.3,
                "dataReliability": 0.1,
            },
        )
        self.assertEqual(contract["displayScale"], 0.95)
        self.assertEqual(contract["maximumScore"], 95.0)
        self.assertEqual(contract["confidence"], "high")
        self.assertTrue(contract["calibrated"])


if __name__ == "__main__":
    unittest.main()
