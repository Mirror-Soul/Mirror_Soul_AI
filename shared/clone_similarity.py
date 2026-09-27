from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


CLONE_SIMILARITY_CALCULATION_VERSION = "clone-similarity-v1"
CLONE_SIMILARITY_FACE_WEIGHT = 0.30
CLONE_SIMILARITY_VOICE_WEIGHT = 0.30
CLONE_SIMILARITY_PROFILE_WEIGHT = 0.30
CLONE_SIMILARITY_DATA_RELIABILITY_WEIGHT = 0.10
CLONE_SIMILARITY_DISPLAY_SCALE = 0.95
CLONE_SIMILARITY_MAX_SCORE = 95.0


@dataclass(frozen=True)
class OverallCloneSimilarity:
    total_score: float
    raw_score: float
    face_score: float | None
    voice_score: float | None
    profile_score: float | None
    data_reliability_score: float | None
    penalty_score: float
    face_weight: float
    voice_weight: float
    profile_weight: float
    data_reliability_weight: float

    @property
    def complete(self) -> bool:
        return all(
            score is not None
            for score in (
                self.face_score,
                self.voice_score,
                self.profile_score,
                self.data_reliability_score,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "calculationVersion": CLONE_SIMILARITY_CALCULATION_VERSION,
            "totalScore": self.total_score,
            "rawScore": self.raw_score,
            "faceScore": self.face_score,
            "voiceScore": self.voice_score,
            "profileScore": self.profile_score,
            "dataReliabilityScore": self.data_reliability_score,
            "penaltyScore": self.penalty_score,
            "complete": self.complete,
            "displayScale": CLONE_SIMILARITY_DISPLAY_SCALE,
            "maximumScore": CLONE_SIMILARITY_MAX_SCORE,
            "weights": {
                "face": self.face_weight,
                "voice": self.voice_weight,
                "profile": self.profile_weight,
                "dataReliability": self.data_reliability_weight,
            },
        }


def calculate_overall_clone_similarity(
    *,
    face_score: float | None,
    voice_score: float | None,
    profile_score: float | None = None,
    data_reliability_score: float | None = None,
    penalty_score: float = 0.0,
    face_weight: float = CLONE_SIMILARITY_FACE_WEIGHT,
    voice_weight: float = CLONE_SIMILARITY_VOICE_WEIGHT,
    profile_weight: float = CLONE_SIMILARITY_PROFILE_WEIGHT,
    data_reliability_weight: float = CLONE_SIMILARITY_DATA_RELIABILITY_WEIGHT,
) -> OverallCloneSimilarity:
    normalized_face = _normalize_optional_score(face_score)
    normalized_voice = _normalize_optional_score(voice_score)
    normalized_profile = _normalize_optional_score(profile_score)
    normalized_reliability = _normalize_optional_score(data_reliability_score)
    normalized_penalty = _normalize_penalty(penalty_score)
    components = (
        normalized_face,
        normalized_voice,
        normalized_profile,
        normalized_reliability,
    )
    if all(component is None for component in components):
        raise ValueError("at least one clone similarity component is required")

    weights = (
        face_weight,
        voice_weight,
        profile_weight,
        data_reliability_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("clone similarity weights cannot be negative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("clone similarity weights must add up to 1.0")

    # Missing components intentionally contribute zero. Re-normalizing the
    # available weights would make an incomplete clone look fully trained.
    weighted_score = sum(
        (score or 0.0) * weight
        for score, weight in zip(components, weights, strict=True)
    )
    raw_score = max(0.0, min(weighted_score - normalized_penalty, 100.0))
    total_score = min(
        raw_score * CLONE_SIMILARITY_DISPLAY_SCALE,
        CLONE_SIMILARITY_MAX_SCORE,
    )

    return OverallCloneSimilarity(
        total_score=_round_display_score(total_score),
        raw_score=_round_component_score(raw_score),
        face_score=normalized_face,
        voice_score=normalized_voice,
        profile_score=normalized_profile,
        data_reliability_score=normalized_reliability,
        penalty_score=normalized_penalty,
        face_weight=float(face_weight),
        voice_weight=float(voice_weight),
        profile_weight=float(profile_weight),
        data_reliability_weight=float(data_reliability_weight),
    )


def face_similarity_component_contract(
    face_similarity: dict[str, object] | None,
) -> dict[str, object] | None:
    if not face_similarity or face_similarity.get("score") is None:
        return None
    face_score = _normalize_optional_score(float(face_similarity["score"]))
    return {
        "calculationVersion": CLONE_SIMILARITY_CALCULATION_VERSION,
        "faceScore": face_score,
        "weights": {
            "face": CLONE_SIMILARITY_FACE_WEIGHT,
            "voice": CLONE_SIMILARITY_VOICE_WEIGHT,
            "profile": CLONE_SIMILARITY_PROFILE_WEIGHT,
            "dataReliability": CLONE_SIMILARITY_DATA_RELIABILITY_WEIGHT,
        },
        "displayScale": CLONE_SIMILARITY_DISPLAY_SCALE,
        "maximumScore": CLONE_SIMILARITY_MAX_SCORE,
        "confidence": face_similarity.get("confidence"),
        "calibrationVersion": face_similarity.get("calibrationVersion"),
        "calibrated": bool(face_similarity.get("calibrated", False)),
    }


def _normalize_optional_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _round_component_score(float(value))


def _normalize_penalty(value: float) -> float:
    return _round_component_score(float(value))


def _round_component_score(value: float) -> float:
    return round(max(0.0, min(value, 100.0)), 2)


def _round_display_score(value: float) -> float:
    bounded = max(0.0, min(value, CLONE_SIMILARITY_MAX_SCORE))
    return float(
        Decimal(str(bounded)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    )
