from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv

from model_training.face_training.face_similarity import (
    FaceSimilarityResult,
    InsightFaceEncoder,
    evaluate_face_similarity,
    face_similarity_config_from_env,
)
from model_training.face_training.liveportrait_runner import (
    LivePortraitConfig,
    LivePortraitResult,
    run_liveportrait,
)
from model_training.face_training.worker import _liveportrait_config


@dataclass(frozen=True)
class FaceVariant:
    driving_multiplier: float
    crop_scale: float

    @property
    def name(self) -> str:
        multiplier = f"{self.driving_multiplier:.2f}".replace(".", "")
        scale = f"{self.crop_scale:.2f}".replace(".", "")
        return f"multiplier-{multiplier}-scale-{scale}"


@dataclass(frozen=True)
class FaceVariantResult:
    variant: FaceVariant
    liveportrait: LivePortraitResult
    similarity: FaceSimilarityResult

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.variant.name,
            "drivingMultiplier": self.variant.driving_multiplier,
            "cropScale": self.variant.crop_scale,
            "livePortrait": self.liveportrait.to_dict(),
            "faceSimilarity": self.similarity.to_dict(),
        }


@dataclass(frozen=True)
class FaceVariantSweepResult:
    best: FaceVariantResult
    candidates: tuple[FaceVariantResult, ...]
    manifest_path: Path
    best_output_path: Path
    best_comparison_path: Path | None


LivePortraitRunner = Callable[..., LivePortraitResult]
SimilarityEvaluator = Callable[..., FaceSimilarityResult]


def run_variant_sweep(
    *,
    source_path: Path,
    driving_path: Path,
    reference_images: Sequence[Path],
    output_dir: Path,
    variants: Sequence[FaceVariant],
    base_config: LivePortraitConfig,
    liveportrait_runner: LivePortraitRunner = run_liveportrait,
    similarity_evaluator: SimilarityEvaluator = evaluate_face_similarity,
) -> FaceVariantSweepResult:
    if not variants:
        raise ValueError("at least one face variant is required")
    if not reference_images:
        raise ValueError("at least one reference image is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, variant in enumerate(variants, start=1):
        print(
            "[FACE_VARIANT] started: "
            f"candidate={index}/{len(variants)} name={variant.name} ",
            f"multiplier={variant.driving_multiplier} scale={variant.crop_scale}",
            flush=True,
        )
        liveportrait = liveportrait_runner(
            source_path,
            driving_path,
            output_dir / variant.name,
            config=replace(
                base_config,
                driving_multiplier=variant.driving_multiplier,
                crop_scale=variant.crop_scale,
            ),
        )
        similarity = similarity_evaluator(
            reference_images=reference_images,
            generated_video=liveportrait.output_path,
            driving_video=driving_path,
        )
        candidate = FaceVariantResult(
            variant=variant,
            liveportrait=liveportrait,
            similarity=similarity,
        )
        candidates.append(candidate)
        print(
            "[FACE_VARIANT] completed: "
            f"name={variant.name} score={similarity.score:.2f} "
            f"identity={similarity.identity_score:.2f} "
            f"render={similarity.render_quality_score:.2f} "
            f"stability={similarity.temporal_consistency:.4f}",
            flush=True,
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.similarity.score,
            item.similarity.render_quality_score,
            item.similarity.identity_score,
        ),
        reverse=True,
    )
    best = ranked[0]
    best_output_path = output_dir / "best.mp4"
    shutil.copy2(best.liveportrait.output_path, best_output_path)
    best_comparison_path = None
    if best.liveportrait.comparison_path is not None:
        best_comparison_path = output_dir / "best-concat.mp4"
        shutil.copy2(best.liveportrait.comparison_path, best_comparison_path)

    manifest_path = output_dir / "variant-sweep.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bestCandidate": best.variant.name,
                "bestOutputPath": str(best_output_path),
                "bestComparisonPath": (
                    str(best_comparison_path)
                    if best_comparison_path is not None
                    else None
                ),
                "ranking": [candidate.to_dict() for candidate in ranked],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[FACE_VARIANT] best selected: "
        f"name={best.variant.name} score={best.similarity.score:.2f} "
        f"output={best_output_path}",
        flush=True,
    )
    return FaceVariantSweepResult(
        best=best,
        candidates=tuple(ranked),
        manifest_path=manifest_path,
        best_output_path=best_output_path,
        best_comparison_path=best_comparison_path,
    )


def variants_from_grid(
    multipliers: Sequence[float],
    crop_scales: Sequence[float],
) -> tuple[FaceVariant, ...]:
    variants = tuple(
        FaceVariant(driving_multiplier=multiplier, crop_scale=scale)
        for multiplier in multipliers
        for scale in crop_scales
    )
    if any(variant.driving_multiplier <= 0 for variant in variants):
        raise ValueError("driving multipliers must be positive")
    if any(variant.crop_scale <= 0 for variant in variants):
        raise ValueError("crop scales must be positive")
    return variants


def inputs_from_manifest(
    manifest_path: Path,
) -> tuple[Path, Path, tuple[Path, ...]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    liveportrait = data.get("livePortrait")
    if not liveportrait:
        raise ValueError("manifest does not contain a LivePortrait result")
    references = tuple(
        Path(frame["path"])
        for video in data["videos"]
        for frame in video["frameSelection"]["frames"]
        if frame["accepted"]
    )
    if not references:
        raise ValueError("manifest does not contain accepted reference frames")
    return (
        Path(liveportrait["sourcePath"]),
        Path(liveportrait["drivingPath"]),
        references,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate and rank multiple LivePortrait face variants.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=(0.65, 0.75, 0.85),
    )
    parser.add_argument(
        "--crop-scales",
        nargs="+",
        type=float,
        default=(2.5, 2.7),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    source, driving, references = inputs_from_manifest(manifest_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else manifest_path.parent / "outputs" / "variant-sweep"
    )
    similarity_config = face_similarity_config_from_env()
    shared_face_encoder = InsightFaceEncoder(similarity_config)

    def evaluate_variant(**kwargs: Any) -> FaceSimilarityResult:
        return evaluate_face_similarity(
            **kwargs,
            config=similarity_config,
            face_encoder=shared_face_encoder,
        )

    result = run_variant_sweep(
        source_path=source,
        driving_path=driving,
        reference_images=references,
        output_dir=output_dir,
        variants=variants_from_grid(args.multipliers, args.crop_scales),
        base_config=_liveportrait_config(),
        similarity_evaluator=evaluate_variant,
    )
    print(f"[FACE_VARIANT] manifest={result.manifest_path}", flush=True)


if __name__ == "__main__":
    main()
