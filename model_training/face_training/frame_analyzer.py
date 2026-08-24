from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


class FaceFrameAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectedFace:
    x: int
    y: int
    width: int
    height: int
    view: str = "front"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameQualityConfig:
    min_sharpness: float = 40.0
    min_brightness: float = 40.0
    max_brightness: float = 215.0
    min_contrast: float = 18.0
    min_face_coverage: float = 0.05
    max_face_coverage: float = 0.70
    max_center_offset: float = 0.55


@dataclass(frozen=True)
class FrameAnalysis:
    path: Path
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    face_count: int
    primary_face: DetectedFace | None
    face_coverage: float
    center_offset: float
    view: str
    quality_score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "sharpness": round(self.sharpness, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "faceCount": self.face_count,
            "primaryFace": (
                self.primary_face.to_dict() if self.primary_face else None
            ),
            "faceCoverage": round(self.face_coverage, 4),
            "centerOffset": round(self.center_offset, 4),
            "view": self.view,
            "qualityScore": round(self.quality_score, 2),
            "accepted": self.accepted,
            "rejectionReasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class FrameSelectionResult:
    frames: tuple[FrameAnalysis, ...]
    selected_source_path: Path | None
    representatives: dict[str, Path]

    @property
    def accepted_count(self) -> int:
        return sum(frame.accepted for frame in self.frames)

    @property
    def rejected_count(self) -> int:
        return len(self.frames) - self.accepted_count

    @property
    def quality_gate_passed(self) -> bool:
        return self.selected_source_path is not None and self.accepted_count >= 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectedSourcePath": (
                str(self.selected_source_path)
                if self.selected_source_path is not None
                else None
            ),
            "representatives": {
                view: str(path) for view, path in sorted(self.representatives.items())
            },
            "acceptedCount": self.accepted_count,
            "rejectedCount": self.rejected_count,
            "qualityGatePassed": self.quality_gate_passed,
            "frames": [frame.to_dict() for frame in self.frames],
        }


GrayImageLoader = Callable[[Path], np.ndarray]
FaceDetector = Callable[[np.ndarray], list[DetectedFace]]


def analyze_face_frames(
    frame_paths: Iterable[Path],
    *,
    config: FrameQualityConfig | None = None,
    image_loader: GrayImageLoader | None = None,
    face_detector: FaceDetector | None = None,
) -> FrameSelectionResult:
    quality_config = config or FrameQualityConfig()
    loader = image_loader or _load_grayscale_with_opencv
    detector = face_detector or OpenCvHaarFaceDetector()

    analyses = []
    for path in frame_paths:
        gray_image = loader(path)
        if gray_image.ndim != 2 or gray_image.size == 0:
            raise FaceFrameAnalysisError(
                f"Expected a non-empty grayscale frame: {path}"
            )
        analyses.append(
            calculate_frame_quality(
                path,
                gray_image,
                detector(gray_image),
                config=quality_config,
            )
        )
    return select_representative_frames(analyses)


def calculate_frame_quality(
    path: Path,
    gray_image: np.ndarray,
    faces: list[DetectedFace],
    *,
    config: FrameQualityConfig | None = None,
) -> FrameAnalysis:
    quality_config = config or FrameQualityConfig()
    height, width = gray_image.shape
    brightness = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))
    sharpness = _laplacian_variance(gray_image)
    primary_face = max(faces, key=lambda face: face.width * face.height, default=None)

    face_coverage = 0.0
    center_offset = 1.0
    view = "unknown"
    if primary_face is not None:
        face_coverage = (
            primary_face.width * primary_face.height / float(width * height)
        )
        face_center_x = primary_face.x + primary_face.width / 2.0
        face_center_y = primary_face.y + primary_face.height / 2.0
        horizontal_offset = abs(face_center_x - width / 2.0) / (width / 2.0)
        vertical_offset = abs(face_center_y - height / 2.0) / (height / 2.0)
        center_offset = min(1.0, (horizontal_offset**2 + vertical_offset**2) ** 0.5)
        view = primary_face.view

    reasons = []
    if not faces:
        reasons.append("face_not_detected")
    elif len(faces) > 1:
        reasons.append("multiple_faces_detected")
    if sharpness < quality_config.min_sharpness:
        reasons.append("too_blurry")
    if brightness < quality_config.min_brightness:
        reasons.append("too_dark")
    elif brightness > quality_config.max_brightness:
        reasons.append("too_bright")
    if contrast < quality_config.min_contrast:
        reasons.append("low_contrast")
    if primary_face is not None:
        if face_coverage < quality_config.min_face_coverage:
            reasons.append("face_too_small")
        elif face_coverage > quality_config.max_face_coverage:
            reasons.append("face_too_close")
        if center_offset > quality_config.max_center_offset:
            reasons.append("face_off_center")

    return FrameAnalysis(
        path=path,
        width=width,
        height=height,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        face_count=len(faces),
        primary_face=primary_face,
        face_coverage=face_coverage,
        center_offset=center_offset,
        view=view,
        quality_score=_quality_score(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            face_coverage=face_coverage,
            center_offset=center_offset,
            face_detected=primary_face is not None,
        ),
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def select_representative_frames(
    analyses: Iterable[FrameAnalysis],
) -> FrameSelectionResult:
    frames = tuple(analyses)
    accepted = [frame for frame in frames if frame.accepted]
    representatives = {
        view: max(
            (frame for frame in accepted if frame.view == view),
            key=lambda frame: frame.quality_score,
        ).path
        for view in sorted({frame.view for frame in accepted})
    }

    source_candidates = [frame for frame in accepted if frame.view == "front"]
    selected_source = max(
        source_candidates,
        key=lambda frame: frame.quality_score,
        default=None,
    )
    return FrameSelectionResult(
        frames=frames,
        selected_source_path=selected_source.path if selected_source else None,
        representatives=representatives,
    )


class OpenCvHaarFaceDetector:
    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise FaceFrameAnalysisError(
                "OpenCV is required for face frame analysis. "
                "Install requirements-face.txt in the GPU environment."
            ) from exc

        self._cv2 = cv2
        cascade_dir = Path(cv2.data.haarcascades)
        self._frontal = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )
        self._profile = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_profileface.xml")
        )
        if self._frontal.empty() or self._profile.empty():
            raise FaceFrameAnalysisError("OpenCV face detector assets are missing.")

    def __call__(self, gray_image: np.ndarray) -> list[DetectedFace]:
        min_size = max(40, min(gray_image.shape) // 10)
        frontal_boxes = self._frontal.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )
        if len(frontal_boxes):
            return [
                DetectedFace(int(x), int(y), int(width), int(height), "front")
                for x, y, width, height in frontal_boxes
            ]

        profile_boxes = self._profile.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        detections = [
            DetectedFace(int(x), int(y), int(width), int(height), "left_profile")
            for x, y, width, height in profile_boxes
        ]

        flipped = self._cv2.flip(gray_image, 1)
        flipped_boxes = self._profile.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        image_width = gray_image.shape[1]
        detections.extend(
            DetectedFace(
                image_width - int(x) - int(width),
                int(y),
                int(width),
                int(height),
                "right_profile",
            )
            for x, y, width, height in flipped_boxes
        )
        return _deduplicate_faces(detections)


def _load_grayscale_with_opencv(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise FaceFrameAnalysisError(
            "OpenCV is required for face frame analysis. "
            "Install requirements-face.txt in the GPU environment."
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FaceFrameAnalysisError(f"Unable to read extracted frame: {path}")
    return image


def _laplacian_variance(gray_image: np.ndarray) -> float:
    image = gray_image.astype(np.float32)
    if min(image.shape) < 3:
        return 0.0
    center = image[1:-1, 1:-1]
    laplacian = (
        image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(laplacian))


def _quality_score(
    *,
    sharpness: float,
    brightness: float,
    contrast: float,
    face_coverage: float,
    center_offset: float,
    face_detected: bool,
) -> float:
    sharpness_score = _clamp((sharpness - 20.0) / 280.0)
    brightness_score = _clamp(1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = _clamp((contrast - 10.0) / 50.0)
    coverage_score = _clamp(1.0 - abs(face_coverage - 0.25) / 0.35)
    center_score = _clamp(1.0 - center_offset)
    face_score = 1.0 if face_detected else 0.0
    return round(
        100.0
        * (
            0.30 * sharpness_score
            + 0.15 * brightness_score
            + 0.15 * contrast_score
            + 0.20 * coverage_score
            + 0.10 * center_score
            + 0.10 * face_score
        ),
        2,
    )


def _deduplicate_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
    selected = []
    for face in sorted(faces, key=lambda item: item.width * item.height, reverse=True):
        if all(_intersection_over_union(face, other) < 0.45 for other in selected):
            selected.append(face)
    return selected


def _intersection_over_union(first: DetectedFace, second: DetectedFace) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / float(first_area + second_area - intersection)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
