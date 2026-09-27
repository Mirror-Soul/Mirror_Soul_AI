from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from model_training.face_training.frame_analyzer import OpenCvHaarFaceDetector


class FaceSimilarityUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceObservation:
    descriptor: np.ndarray
    sharpness: float
    geometry: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FaceSimilarityConfig:
    sample_count: int = 16
    appearance_low: float = 0.35
    appearance_high: float = 0.92
    max_score: float = 100.0
    min_detection_rate: float = 0.75
    source_preservation_weight: float = 0.40
    render_quality_weight: float = 0.60
    min_temporal_consistency: float = 0.60
    stability_floor: float = 0.70
    evaluator_name: str = "opencv-appearance-v1"
    calibration_version: str = "provisional-v1"
    calibrated: bool = False


@dataclass(frozen=True)
class FaceSimilarityResult:
    score: float
    source_preservation_score: float
    render_quality_score: float
    appearance_similarity: float
    aligned_appearance_similarity: float | None
    gallery_appearance_similarity: float
    detection_rate: float
    temporal_consistency: float
    geometry_consistency: float
    sharpness_retention: float
    stability_factor: float
    evaluated_frame_count: int
    detected_frame_count: int
    aligned_frame_count: int
    reference_count: int
    confidence: str
    evaluator_name: str
    provider: str
    calibration_version: str
    calibrated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "sourcePreservationScore": self.source_preservation_score,
            "renderQualityScore": self.render_quality_score,
            "appearanceSimilarity": self.appearance_similarity,
            "alignedAppearanceSimilarity": self.aligned_appearance_similarity,
            "galleryAppearanceSimilarity": self.gallery_appearance_similarity,
            "detectionRate": self.detection_rate,
            "temporalConsistency": self.temporal_consistency,
            "geometryConsistency": self.geometry_consistency,
            "sharpnessRetention": self.sharpness_retention,
            "stabilityFactor": self.stability_factor,
            "evaluatedFrameCount": self.evaluated_frame_count,
            "detectedFrameCount": self.detected_frame_count,
            "alignedFrameCount": self.aligned_frame_count,
            "referenceCount": self.reference_count,
            "confidence": self.confidence,
            "evaluatorName": self.evaluator_name,
            "provider": self.provider,
            "calibrationVersion": self.calibration_version,
            "calibrated": self.calibrated,
        }


FaceEncoder = Callable[[np.ndarray], FaceObservation | None]
VideoSampler = Callable[[Path, int], list[np.ndarray]]
ImageLoader = Callable[[Path], np.ndarray]


def evaluate_face_similarity(
    *,
    reference_images: Sequence[Path],
    generated_video: Path,
    driving_video: Path | None = None,
    config: FaceSimilarityConfig | None = None,
    face_encoder: FaceEncoder | None = None,
    video_sampler: VideoSampler | None = None,
    image_loader: ImageLoader | None = None,
) -> FaceSimilarityResult:
    scoring_config = config or face_similarity_config_from_env()
    _validate_config(scoring_config)
    if not reference_images:
        raise FaceSimilarityUnavailable("no reference face images provided")

    encoder = face_encoder or OpenCvAppearanceEncoder()
    provider = "injected" if face_encoder is not None else "opencv-cpu"
    load_image = image_loader or _load_image
    sample_video = video_sampler or _sample_video_frames

    reference_observations = [
        observation
        for path in reference_images
        if (observation := encoder(load_image(path))) is not None
    ]
    if not reference_observations:
        raise FaceSimilarityUnavailable("no face detected in reference images")

    generated_frames = sample_video(generated_video, scoring_config.sample_count)
    if not generated_frames:
        raise FaceSimilarityUnavailable("generated video has no readable frames")
    generated_observations = [encoder(frame) for frame in generated_frames]

    driving_observations: list[FaceObservation | None] | None = None
    if driving_video is not None:
        driving_frames = sample_video(driving_video, len(generated_frames))
        if driving_frames:
            driving_observations = [encoder(frame) for frame in driving_frames]

    return score_face_observations(
        reference_observations=reference_observations,
        generated_observations=generated_observations,
        driving_observations=driving_observations,
        config=scoring_config,
        provider=provider,
    )


def score_face_observations(
    *,
    reference_observations: Sequence[FaceObservation],
    generated_observations: Sequence[FaceObservation | None],
    driving_observations: Sequence[FaceObservation | None] | None = None,
    config: FaceSimilarityConfig | None = None,
    provider: str = "injected",
) -> FaceSimilarityResult:
    scoring_config = config or FaceSimilarityConfig()
    _validate_config(scoring_config)
    if not reference_observations:
        raise FaceSimilarityUnavailable("no usable reference face observations")
    if not generated_observations:
        raise FaceSimilarityUnavailable("no generated face observations")

    references = [_normalized(item.descriptor) for item in reference_observations]
    detected_generated = [
        (index, item)
        for index, item in enumerate(generated_observations)
        if item is not None
    ]
    if not detected_generated:
        raise FaceSimilarityUnavailable("no face detected in generated video")

    gallery_similarities = [
        max(_cosine(item.descriptor, reference) for reference in references)
        for _, item in detected_generated
    ]
    gallery_similarity = _robust_similarity(gallery_similarities)

    aligned_similarities = []
    sharpness_ratios = []
    if driving_observations is not None:
        for index, generated in detected_generated:
            if index >= len(driving_observations):
                continue
            driving = driving_observations[index]
            if driving is None:
                continue
            aligned_similarities.append(
                _cosine(generated.descriptor, driving.descriptor)
            )
            if driving.sharpness > 1e-6:
                sharpness_ratios.append(
                    _clamp(generated.sharpness / driving.sharpness)
                )
    else:
        reference_sharpness = float(
            np.median([item.sharpness for item in reference_observations])
        )
        if reference_sharpness > 1e-6:
            sharpness_ratios = [
                _clamp(item.sharpness / reference_sharpness)
                for _, item in detected_generated
            ]

    aligned_similarity = (
        _robust_similarity(aligned_similarities)
        if aligned_similarities
        else None
    )
    appearance_similarity = (
        0.65 * aligned_similarity + 0.35 * gallery_similarity
        if aligned_similarity is not None
        else gallery_similarity
    )
    source_preservation_score = _appearance_to_score(
        appearance_similarity,
        scoring_config,
    )

    detection_rate = len(detected_generated) / len(generated_observations)
    temporal_consistency = _temporal_consistency(gallery_similarities)
    geometry_consistency = _geometry_consistency(
        [item.geometry for _, item in detected_generated if item.geometry is not None]
    )
    sharpness_retention = (
        float(np.median(sharpness_ratios)) if sharpness_ratios else 0.0
    )
    render_quality_score = 100.0 * (
        0.35 * detection_rate
        + 0.25 * temporal_consistency
        + 0.25 * sharpness_retention
        + 0.15 * geometry_consistency
    )

    weight_sum = (
        scoring_config.source_preservation_weight
        + scoring_config.render_quality_weight
    )
    blended_score = (
        source_preservation_score * scoring_config.source_preservation_weight
        + render_quality_score * scoring_config.render_quality_weight
    ) / weight_sum

    stability_ratio = min(
        temporal_consistency / scoring_config.min_temporal_consistency,
        1.0,
    )
    stability_factor = scoring_config.stability_floor + (
        1.0 - scoring_config.stability_floor
    ) * stability_ratio
    blended_score *= stability_factor

    coverage_ratio = min(
        detection_rate / scoring_config.min_detection_rate,
        1.0,
    )
    if coverage_ratio < 1.0:
        blended_score *= 0.50 + 0.50 * coverage_ratio

    # Generic image quality must not hide large source-preservation changes.
    blended_score = min(blended_score, source_preservation_score + 20.0)

    return FaceSimilarityResult(
        score=_round_score(min(blended_score, scoring_config.max_score)),
        source_preservation_score=_round_score(source_preservation_score),
        render_quality_score=_round_score(render_quality_score),
        appearance_similarity=round(float(appearance_similarity), 4),
        aligned_appearance_similarity=(
            round(float(aligned_similarity), 4)
            if aligned_similarity is not None
            else None
        ),
        gallery_appearance_similarity=round(float(gallery_similarity), 4),
        detection_rate=round(detection_rate, 4),
        temporal_consistency=round(temporal_consistency, 4),
        geometry_consistency=round(geometry_consistency, 4),
        sharpness_retention=round(sharpness_retention, 4),
        stability_factor=round(stability_factor, 4),
        evaluated_frame_count=len(generated_observations),
        detected_frame_count=len(detected_generated),
        aligned_frame_count=len(aligned_similarities),
        reference_count=len(reference_observations),
        confidence=_confidence(
            detection_rate=detection_rate,
            detected_frame_count=len(detected_generated),
            reference_count=len(reference_observations),
        ),
        evaluator_name=scoring_config.evaluator_name,
        provider=provider,
        calibration_version=scoring_config.calibration_version,
        calibrated=scoring_config.calibrated,
    )


class OpenCvAppearanceEncoder:
    """Builds a non-biometric descriptor from stable upper-face pixels."""

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise FaceSimilarityUnavailable(
                "OpenCV is required for face rendering evaluation"
            ) from exc
        self._cv2 = cv2
        self._detector = OpenCvHaarFaceDetector()

    def __call__(self, image: np.ndarray) -> FaceObservation | None:
        gray = (
            self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        faces = self._detector(gray)
        if not faces:
            return None
        face = max(faces, key=lambda item: item.width * item.height)
        height, width = gray.shape[:2]
        x1 = max(0, face.x)
        y1 = max(0, face.y)
        x2 = min(width, face.x + face.width)
        y2 = min(height, face.y + face.height)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # The mouth is excluded so speaking does not look like identity drift.
        stable_height = max(1, int(crop.shape[0] * 0.68))
        stable_region = crop[:stable_height, :]
        normalized_region = self._cv2.equalizeHist(stable_region)
        resized = self._cv2.resize(
            normalized_region,
            (32, 24),
            interpolation=self._cv2.INTER_AREA,
        ).astype(np.float32)
        resized -= float(np.mean(resized))
        descriptor = _normalized(resized.reshape(-1))

        sharpness = float(
            self._cv2.Laplacian(crop, self._cv2.CV_64F).var()
        )
        geometry = (
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        )
        return FaceObservation(
            descriptor=descriptor,
            sharpness=sharpness,
            geometry=geometry,
        )


def face_similarity_config_from_env() -> FaceSimilarityConfig:
    return FaceSimilarityConfig(
        sample_count=_env_int("FACE_SIMILARITY_SAMPLE_COUNT", 16),
        appearance_low=_env_float("FACE_SIMILARITY_APPEARANCE_LOW", 0.35),
        appearance_high=_env_float("FACE_SIMILARITY_APPEARANCE_HIGH", 0.92),
        max_score=_env_float("FACE_SIMILARITY_COMPONENT_MAX", 100.0),
        min_detection_rate=_env_float(
            "FACE_SIMILARITY_MIN_DETECTION_RATE",
            0.75,
        ),
        source_preservation_weight=_env_float(
            "FACE_SIMILARITY_SOURCE_PRESERVATION_WEIGHT",
            0.40,
        ),
        render_quality_weight=_env_float(
            "FACE_SIMILARITY_RENDER_QUALITY_WEIGHT",
            0.60,
        ),
        min_temporal_consistency=_env_float(
            "FACE_SIMILARITY_MIN_TEMPORAL_CONSISTENCY",
            0.60,
        ),
        stability_floor=_env_float("FACE_SIMILARITY_STABILITY_FLOOR", 0.70),
        evaluator_name=os.getenv(
            "FACE_SIMILARITY_EVALUATOR",
            "opencv-appearance-v1",
        ),
        calibration_version=os.getenv(
            "FACE_SIMILARITY_CALIBRATION_VERSION",
            "provisional-v1",
        ),
        calibrated=_env_bool("FACE_SIMILARITY_CALIBRATED", False),
    )


def _load_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise FaceSimilarityUnavailable("OpenCV is required") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceSimilarityUnavailable(f"unable to read reference image: {path}")
    return image


def _sample_video_frames(path: Path, sample_count: int) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise FaceSimilarityUnavailable("OpenCV is required") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FaceSimilarityUnavailable(f"unable to open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return []
        indices = sorted(
            {
                int(round(value))
                for value in np.linspace(
                    0,
                    frame_count - 1,
                    min(sample_count, frame_count),
                )
            }
        )
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if success and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def _robust_similarity(values: Sequence[float]) -> float:
    if not values:
        raise FaceSimilarityUnavailable("no face similarities available")
    array = np.asarray(values, dtype=np.float32)
    return float(0.70 * np.median(array) + 0.30 * np.percentile(array, 10))


def _temporal_consistency(similarities: Sequence[float]) -> float:
    if len(similarities) < 2:
        return 0.0
    array = np.asarray(similarities, dtype=np.float32)
    spread = float(np.percentile(array, 90) - np.percentile(array, 10))
    return _clamp(1.0 - spread / 0.35)


def _geometry_consistency(
    geometries: Sequence[tuple[float, float, float, float]],
) -> float:
    if len(geometries) < 2:
        return 0.0
    array = np.asarray(geometries, dtype=np.float32)
    center_jitter = float(np.mean(np.std(array[:, :2], axis=0)))
    size_jitter = float(np.mean(np.std(array[:, 2:], axis=0)))
    return _clamp(1.0 - center_jitter / 0.08 - size_jitter / 0.10)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalized(left), _normalized(right)))


def _normalized(descriptor: np.ndarray) -> np.ndarray:
    array = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise FaceSimilarityUnavailable("face appearance descriptor has zero norm")
    return array / norm


def _appearance_to_score(
    appearance_similarity: float,
    config: FaceSimilarityConfig,
) -> float:
    normalized = (appearance_similarity - config.appearance_low) / (
        config.appearance_high - config.appearance_low
    )
    return _clamp(normalized) * config.max_score


def _confidence(
    *,
    detection_rate: float,
    detected_frame_count: int,
    reference_count: int,
) -> str:
    if detection_rate >= 0.90 and detected_frame_count >= 12 and reference_count >= 3:
        return "high"
    if detection_rate >= 0.70 and detected_frame_count >= 8 and reference_count >= 2:
        return "medium"
    return "low"


def _validate_config(config: FaceSimilarityConfig) -> None:
    if config.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if config.appearance_high <= config.appearance_low:
        raise ValueError("appearance_high must be greater than appearance_low")
    if not -1.0 <= config.appearance_low < config.appearance_high <= 1.0:
        raise ValueError("appearance thresholds must be between -1 and 1")
    if not 0.0 < config.max_score <= 100.0:
        raise ValueError("max_score must be between 0 and 100")
    if not 0.0 < config.min_detection_rate <= 1.0:
        raise ValueError("min_detection_rate must be between 0 and 1")
    if (
        config.source_preservation_weight < 0
        or config.render_quality_weight < 0
    ):
        raise ValueError("face quality weights cannot be negative")
    if config.source_preservation_weight + config.render_quality_weight <= 0:
        raise ValueError("at least one face quality weight must be positive")
    if not 0.0 < config.min_temporal_consistency <= 1.0:
        raise ValueError("min_temporal_consistency must be between 0 and 1")
    if not 0.0 <= config.stability_floor <= 1.0:
        raise ValueError("stability_floor must be between 0 and 1")


def _round_score(value: float) -> float:
    return round(max(0.0, min(float(value), 100.0)), 2)


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Measure source preservation and face rendering quality.",
    )
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--driving")
    args = parser.parse_args()

    result = evaluate_face_similarity(
        reference_images=[Path(path) for path in args.reference],
        generated_video=Path(args.generated),
        driving_video=Path(args.driving) if args.driving else None,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
