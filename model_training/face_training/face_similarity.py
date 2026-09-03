from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


class FaceSimilarityUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceObservation:
    embedding: np.ndarray
    sharpness: float


@dataclass(frozen=True)
class FaceSimilarityConfig:
    sample_count: int = 16
    cosine_low: float = 0.30
    cosine_high: float = 0.70
    max_score: float = 95.0
    min_detection_rate: float = 0.75
    identity_weight: float = 0.65
    render_quality_weight: float = 0.35
    min_temporal_consistency: float = 0.60
    stability_floor: float = 0.70
    model_name: str = "buffalo_l"
    model_root: Path = Path("/workspace/mirror-soul-face/insightface")
    providers: tuple[str, ...] = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    calibration_version: str = "provisional-v1"
    calibrated: bool = False


@dataclass(frozen=True)
class FaceSimilarityResult:
    score: float
    identity_score: float
    render_quality_score: float
    cosine_similarity: float
    aligned_cosine_similarity: float | None
    gallery_cosine_similarity: float
    detection_rate: float
    temporal_consistency: float
    sharpness_retention: float
    stability_factor: float
    evaluated_frame_count: int
    detected_frame_count: int
    aligned_frame_count: int
    reference_count: int
    confidence: str
    model_name: str
    provider: str
    calibration_version: str
    calibrated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "identityScore": self.identity_score,
            "renderQualityScore": self.render_quality_score,
            "cosineSimilarity": self.cosine_similarity,
            "alignedCosineSimilarity": self.aligned_cosine_similarity,
            "galleryCosineSimilarity": self.gallery_cosine_similarity,
            "detectionRate": self.detection_rate,
            "temporalConsistency": self.temporal_consistency,
            "sharpnessRetention": self.sharpness_retention,
            "stabilityFactor": self.stability_factor,
            "evaluatedFrameCount": self.evaluated_frame_count,
            "detectedFrameCount": self.detected_frame_count,
            "alignedFrameCount": self.aligned_frame_count,
            "referenceCount": self.reference_count,
            "confidence": self.confidence,
            "modelName": self.model_name,
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

    encoder = face_encoder
    provider = "injected"
    if encoder is None:
        insightface_encoder = InsightFaceEncoder(scoring_config)
        encoder = insightface_encoder
        provider = insightface_encoder.provider

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

    references = [_normalized(item.embedding) for item in reference_observations]
    detected_generated = [
        (index, item)
        for index, item in enumerate(generated_observations)
        if item is not None
    ]
    if not detected_generated:
        raise FaceSimilarityUnavailable("no face detected in generated video")

    gallery_similarities = [
        max(_cosine(item.embedding, reference) for reference in references)
        for _, item in detected_generated
    ]
    gallery_cosine = _robust_similarity(gallery_similarities)

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
                _cosine(generated.embedding, driving.embedding)
            )
            if driving.sharpness > 1e-6:
                sharpness_ratios.append(
                    min(max(generated.sharpness / driving.sharpness, 0.0), 1.0)
                )

    aligned_cosine = (
        _robust_similarity(aligned_similarities)
        if aligned_similarities
        else None
    )
    cosine_similarity = (
        0.65 * aligned_cosine + 0.35 * gallery_cosine
        if aligned_cosine is not None
        else gallery_cosine
    )
    identity_score = _cosine_to_score(cosine_similarity, scoring_config)

    detection_rate = len(detected_generated) / len(generated_observations)
    temporal_consistency = _temporal_consistency(gallery_similarities)
    sharpness_retention = (
        float(np.median(sharpness_ratios)) if sharpness_ratios else 0.0
    )
    render_quality_score = 100.0 * (
        0.50 * detection_rate
        + 0.30 * temporal_consistency
        + 0.20 * sharpness_retention
    )

    weight_sum = (
        scoring_config.identity_weight + scoring_config.render_quality_weight
    )
    blended_score = (
        identity_score * scoring_config.identity_weight
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

    # Rendering quality may refine a matching identity, but it must not make a
    # different person look similar on its own.
    blended_score = min(blended_score, identity_score + 15.0)

    return FaceSimilarityResult(
        score=_round_score(min(blended_score, scoring_config.max_score)),
        identity_score=_round_score(identity_score),
        render_quality_score=_round_score(render_quality_score),
        cosine_similarity=round(float(cosine_similarity), 4),
        aligned_cosine_similarity=(
            round(float(aligned_cosine), 4)
            if aligned_cosine is not None
            else None
        ),
        gallery_cosine_similarity=round(float(gallery_cosine), 4),
        detection_rate=round(detection_rate, 4),
        temporal_consistency=round(temporal_consistency, 4),
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
        model_name=scoring_config.model_name,
        provider=provider,
        calibration_version=scoring_config.calibration_version,
        calibrated=scoring_config.calibrated,
    )


class InsightFaceEncoder:
    def __init__(self, config: FaceSimilarityConfig) -> None:
        if not _env_bool(
            "FACE_SIMILARITY_ACCEPT_INSIGHTFACE_NON_COMMERCIAL_LICENSE",
            False,
        ):
            raise FaceSimilarityUnavailable(
                "InsightFace public model weights require explicit acceptance for "
                "non-commercial research use. Set "
                "FACE_SIMILARITY_ACCEPT_INSIGHTFACE_NON_COMMERCIAL_LICENSE=true "
                "only when that use is appropriate."
            )
        try:
            import cv2
            import insightface
            import onnxruntime as ort
        except ImportError as exc:
            raise FaceSimilarityUnavailable(
                "InsightFace dependencies are missing. Install "
                "requirements-face-similarity.txt."
            ) from exc

        available = set(ort.get_available_providers())
        providers = [name for name in config.providers if name in available]
        if not providers:
            raise FaceSimilarityUnavailable(
                "none of the configured ONNX Runtime providers are available"
            )

        self._cv2 = cv2
        try:
            self._app = insightface.app.FaceAnalysis(
                name=config.model_name,
                root=str(config.model_root),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(
                ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                det_size=(640, 640),
            )
        except Exception as exc:
            raise FaceSimilarityUnavailable(
                f"unable to initialize InsightFace model {config.model_name}: {exc}"
            ) from exc
        self.provider = _active_provider(self._app, providers)

    def __call__(self, image: np.ndarray) -> FaceObservation | None:
        faces = self._app.get(image)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda item: (
                float(item.bbox[2] - item.bbox[0])
                * float(item.bbox[3] - item.bbox[1])
            ),
        )
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None
        x1, y1, x2, y2 = _bounded_box(face.bbox, image.shape)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2GRAY)
        sharpness = float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())
        return FaceObservation(
            embedding=_normalized(np.asarray(embedding, dtype=np.float32)),
            sharpness=sharpness,
        )


def face_similarity_config_from_env() -> FaceSimilarityConfig:
    providers = tuple(
        item.strip()
        for item in os.getenv(
            "FACE_SIMILARITY_PROVIDERS",
            "CUDAExecutionProvider,CPUExecutionProvider",
        ).split(",")
        if item.strip()
    )
    return FaceSimilarityConfig(
        sample_count=_env_int("FACE_SIMILARITY_SAMPLE_COUNT", 16),
        cosine_low=_env_float("FACE_SIMILARITY_COSINE_LOW", 0.30),
        cosine_high=_env_float("FACE_SIMILARITY_COSINE_HIGH", 0.70),
        max_score=_env_float("FACE_SIMILARITY_MAX_SCORE", 95.0),
        min_detection_rate=_env_float(
            "FACE_SIMILARITY_MIN_DETECTION_RATE",
            0.75,
        ),
        identity_weight=_env_float("FACE_SIMILARITY_IDENTITY_WEIGHT", 0.65),
        render_quality_weight=_env_float(
            "FACE_SIMILARITY_RENDER_QUALITY_WEIGHT",
            0.35,
        ),
        min_temporal_consistency=_env_float(
            "FACE_SIMILARITY_MIN_TEMPORAL_CONSISTENCY",
            0.60,
        ),
        stability_floor=_env_float("FACE_SIMILARITY_STABILITY_FLOOR", 0.70),
        model_name=os.getenv("FACE_SIMILARITY_MODEL", "buffalo_l"),
        model_root=Path(
            os.getenv(
                "FACE_SIMILARITY_MODEL_ROOT",
                "/workspace/mirror-soul-face/insightface",
            )
        ),
        providers=providers,
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


def _active_provider(app: object, configured: Sequence[str]) -> str:
    models = getattr(app, "models", {})
    recognition = models.get("recognition") if isinstance(models, dict) else None
    session = getattr(recognition, "session", None)
    get_providers = getattr(session, "get_providers", None)
    if callable(get_providers):
        active = get_providers()
        if active:
            return str(active[0])
    return configured[0]


def _bounded_box(bbox: Sequence[float], shape: Sequence[int]) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), width))
    y2 = max(y1 + 1, min(int(bbox[3]), height))
    return x1, y1, x2, y2


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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalized(left), _normalized(right)))


def _normalized(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise FaceSimilarityUnavailable("face embedding has zero norm")
    return array / norm


def _cosine_to_score(
    cosine_similarity: float,
    config: FaceSimilarityConfig,
) -> float:
    normalized = (cosine_similarity - config.cosine_low) / (
        config.cosine_high - config.cosine_low
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
    if config.cosine_high <= config.cosine_low:
        raise ValueError("cosine_high must be greater than cosine_low")
    if not -1.0 <= config.cosine_low < config.cosine_high <= 1.0:
        raise ValueError("cosine thresholds must be between -1 and 1")
    if not 0.0 < config.max_score <= 100.0:
        raise ValueError("max_score must be between 0 and 100")
    if not 0.0 < config.min_detection_rate <= 1.0:
        raise ValueError("min_detection_rate must be between 0 and 1")
    if config.identity_weight < 0 or config.render_quality_weight < 0:
        raise ValueError("similarity weights cannot be negative")
    if config.identity_weight + config.render_quality_weight <= 0:
        raise ValueError("at least one similarity weight must be positive")
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
        description="Measure generated face identity and rendering similarity.",
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
