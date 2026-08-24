import unittest
from pathlib import Path

import numpy as np

from model_training.face_training.frame_analyzer import (
    DetectedFace,
    FrameAnalysis,
    FrameQualityConfig,
    analyze_face_frames,
    calculate_frame_quality,
    select_representative_frames,
)


class FrameAnalyzerTests(unittest.TestCase):
    def test_rejects_a_frame_without_a_detected_face(self) -> None:
        image = _checkerboard()

        analysis = calculate_frame_quality(
            Path("no-face.jpg"),
            image,
            [],
            config=_lenient_config(),
        )

        self.assertFalse(analysis.accepted)
        self.assertIn("face_not_detected", analysis.rejection_reasons)

    def test_rejects_a_blurry_frame(self) -> None:
        image = np.full((100, 100), 128, dtype=np.uint8)

        analysis = calculate_frame_quality(
            Path("blurry.jpg"),
            image,
            [DetectedFace(25, 20, 50, 60)],
            config=FrameQualityConfig(
                min_sharpness=1.0,
                min_brightness=0.0,
                max_brightness=255.0,
                min_contrast=0.0,
                min_face_coverage=0.01,
                max_face_coverage=0.90,
                max_center_offset=1.0,
            ),
        )

        self.assertFalse(analysis.accepted)
        self.assertIn("too_blurry", analysis.rejection_reasons)

    def test_prefers_front_frame_for_liveportrait_source(self) -> None:
        front = _analysis("front.jpg", "front", 72.0)
        profile = _analysis("profile.jpg", "left_profile", 94.0)

        result = select_representative_frames([profile, front])

        self.assertEqual(result.selected_source_path, Path("front.jpg"))
        self.assertEqual(result.representatives["front"], Path("front.jpg"))
        self.assertEqual(
            result.representatives["left_profile"], Path("profile.jpg")
        )

    def test_does_not_use_profile_as_source_when_front_is_missing(self) -> None:
        profile = _analysis("profile.jpg", "left_profile", 94.0)

        result = select_representative_frames([profile])

        self.assertIsNone(result.selected_source_path)
        self.assertFalse(result.quality_gate_passed)

    def test_analyzes_frames_with_injected_image_and_face_adapters(self) -> None:
        paths = [Path("front.jpg"), Path("side.jpg")]
        images = {path: _checkerboard() for path in paths}
        calls = 0

        def detector(_: np.ndarray) -> list[DetectedFace]:
            nonlocal calls
            view = "front" if calls == 0 else "right_profile"
            calls += 1
            return [DetectedFace(25, 20, 50, 60, view)]

        result = analyze_face_frames(
            paths,
            config=_lenient_config(),
            image_loader=images.__getitem__,
            face_detector=detector,
        )

        self.assertEqual(result.accepted_count, 2)
        self.assertEqual(result.selected_source_path, Path("front.jpg"))
        self.assertEqual(set(result.representatives), {"front", "right_profile"})


def _checkerboard() -> np.ndarray:
    rows, columns = np.indices((100, 100))
    return ((rows + columns) % 2 * 180 + 40).astype(np.uint8)


def _lenient_config() -> FrameQualityConfig:
    return FrameQualityConfig(
        min_sharpness=0.0,
        min_brightness=0.0,
        max_brightness=255.0,
        min_contrast=0.0,
        min_face_coverage=0.01,
        max_face_coverage=0.90,
        max_center_offset=1.0,
    )


def _analysis(path: str, view: str, score: float) -> FrameAnalysis:
    return FrameAnalysis(
        path=Path(path),
        width=100,
        height=100,
        sharpness=100.0,
        brightness=128.0,
        contrast=40.0,
        face_count=1,
        primary_face=DetectedFace(25, 20, 50, 60, view),
        face_coverage=0.30,
        center_offset=0.0,
        view=view,
        quality_score=score,
        accepted=True,
        rejection_reasons=(),
    )


if __name__ == "__main__":
    unittest.main()
