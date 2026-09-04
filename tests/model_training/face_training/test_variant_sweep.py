import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

from model_training.face_training.face_similarity import FaceSimilarityResult
from model_training.face_training.liveportrait_runner import (
    LivePortraitConfig,
    LivePortraitResult,
)
from model_training.face_training.variant_sweep import (
    FaceVariant,
    inputs_from_manifest,
    run_variant_sweep,
    variants_from_grid,
)


def _similarity(score: float) -> FaceSimilarityResult:
    return FaceSimilarityResult(
        score=score,
        identity_score=90.0,
        render_quality_score=score,
        cosine_similarity=0.7,
        aligned_cosine_similarity=0.7,
        gallery_cosine_similarity=0.7,
        detection_rate=1.0,
        temporal_consistency=0.8,
        sharpness_retention=0.9,
        stability_factor=1.0,
        evaluated_frame_count=16,
        detected_frame_count=16,
        aligned_frame_count=16,
        reference_count=3,
        confidence="high",
        model_name="buffalo_l",
        provider="injected",
        calibration_version="test-v1",
        calibrated=True,
    )


class FaceVariantSweepTests(unittest.TestCase):
    def test_generates_ranks_and_copies_best_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            driving = root / "driving.mp4"
            reference = root / "reference.jpg"
            for path in (source, driving, reference):
                path.write_bytes(b"input")

            scores = {
                "multiplier-065-scale-250": 72.0,
                "multiplier-075-scale-250": 84.0,
            }

            def fake_runner(source_path, driving_path, output_dir, *, config):
                output_dir.mkdir(parents=True)
                output = output_dir / "result.mp4"
                comparison = output_dir / "result_concat.mp4"
                output.write_bytes(output_dir.name.encode())
                comparison.write_bytes(b"comparison")
                return LivePortraitResult(
                    source_path=source_path,
                    driving_path=driving_path,
                    output_path=output,
                    comparison_path=comparison,
                    log_path=output_dir / "liveportrait.log",
                    duration_seconds=1.0,
                    command=(str(config.driving_multiplier), str(config.crop_scale)),
                )

            def fake_evaluator(**kwargs):
                name = kwargs["generated_video"].parent.name
                return _similarity(scores[name])

            result = run_variant_sweep(
                source_path=source,
                driving_path=driving,
                reference_images=[reference],
                output_dir=root / "variants",
                variants=(
                    FaceVariant(0.65, 2.5),
                    FaceVariant(0.75, 2.5),
                ),
                base_config=LivePortraitConfig(
                    repository_dir=root,
                    python_binary=Path(sys.executable),
                ),
                liveportrait_runner=fake_runner,
                similarity_evaluator=fake_evaluator,
            )

            self.assertEqual(result.best.variant.name, "multiplier-075-scale-250")
            self.assertEqual(result.best_output_path.read_bytes(), b"multiplier-075-scale-250")
            self.assertTrue(result.best_comparison_path.is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["bestCandidate"], "multiplier-075-scale-250")
            self.assertEqual(len(manifest["ranking"]), 2)

    def test_builds_cartesian_variant_grid(self) -> None:
        variants = variants_from_grid((0.65, 0.75), (2.5, 2.7))

        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[-1], FaceVariant(0.75, 2.7))

    def test_reads_source_driving_and_references_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "livePortrait": {
                            "sourcePath": "/tmp/source.jpg",
                            "drivingPath": "/tmp/driving.mp4",
                        },
                        "videos": [
                            {
                                "frameSelection": {
                                    "frames": [
                                        {"path": "/tmp/front.jpg", "accepted": True},
                                        {"path": "/tmp/blur.jpg", "accepted": False},
                                    ]
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            source, driving, references = inputs_from_manifest(manifest_path)

            self.assertEqual(source, Path("/tmp/source.jpg"))
            self.assertEqual(driving, Path("/tmp/driving.mp4"))
            self.assertEqual(references, (Path("/tmp/front.jpg"),))


if __name__ == "__main__":
    unittest.main()
