import unittest

import numpy as np

from model_training.face_training.face_similarity import (
    FaceObservation,
    FaceSimilarityConfig,
    FaceSimilarityUnavailable,
    score_face_observations,
)


def _observation(values, sharpness=100.0):
    return FaceObservation(
        embedding=np.asarray(values, dtype=np.float32),
        sharpness=sharpness,
    )


class FaceSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FaceSimilarityConfig(
            sample_count=16,
            cosine_low=0.30,
            cosine_high=0.70,
            max_score=95.0,
        )
        self.references = [
            _observation([1.0, 0.0]),
            _observation([0.99, 0.01]),
            _observation([0.98, -0.02]),
        ]

    def test_matching_identity_receives_high_bounded_score(self) -> None:
        generated = [_observation([1.0, 0.0], 90.0) for _ in range(16)]
        driving = [_observation([1.0, 0.0], 100.0) for _ in range(16)]

        result = score_face_observations(
            reference_observations=self.references,
            generated_observations=generated,
            driving_observations=driving,
            config=self.config,
            provider="CUDAExecutionProvider",
        )

        self.assertEqual(result.score, 95.0)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.provider, "CUDAExecutionProvider")
        self.assertFalse(result.calibrated)
        self.assertEqual(result.detected_frame_count, 16)

    def test_different_identity_receives_low_score(self) -> None:
        generated = [_observation([0.0, 1.0]) for _ in range(16)]
        driving = [_observation([1.0, 0.0]) for _ in range(16)]

        result = score_face_observations(
            reference_observations=self.references,
            generated_observations=generated,
            driving_observations=driving,
            config=self.config,
        )

        self.assertLess(result.score, 20.0)
        self.assertEqual(result.identity_score, 0.0)

    def test_low_detection_rate_reduces_score_and_confidence(self) -> None:
        generated = [
            *[_observation([1.0, 0.0]) for _ in range(4)],
            *[None for _ in range(12)],
        ]
        driving = [_observation([1.0, 0.0]) for _ in range(16)]

        result = score_face_observations(
            reference_observations=self.references,
            generated_observations=generated,
            driving_observations=driving,
            config=self.config,
        )

        self.assertEqual(result.detection_rate, 0.25)
        self.assertEqual(result.confidence, "low")
        self.assertLess(result.score, 70.0)

    def test_rejects_video_without_detected_faces(self) -> None:
        with self.assertRaisesRegex(
            FaceSimilarityUnavailable,
            "no face detected in generated video",
        ):
            score_face_observations(
                reference_observations=self.references,
                generated_observations=[None, None],
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
