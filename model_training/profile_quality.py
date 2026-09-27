from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


_PLACEHOLDER_ANSWERS = {
    "...",
    "asdf",
    "qwer",
    "test",
    "테스트",
    "패스",
    "ㅇㅇ",
    "ㄴㄴ",
}


@dataclass(frozen=True)
class ProfileQualityScore:
    profile_score: float
    data_reliability_score: float
    penalty_score: float
    meaningful_answer_count: int
    duplicate_answer_count: int
    placeholder_answer_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "profileScore": self.profile_score,
            "dataReliabilityScore": self.data_reliability_score,
            "penaltyScore": self.penalty_score,
            "meaningfulAnswerCount": self.meaningful_answer_count,
            "duplicateAnswerCount": self.duplicate_answer_count,
            "placeholderAnswerCount": self.placeholder_answer_count,
        }


def evaluate_profile_quality(
    *,
    age: int | None,
    gender: str | None,
    mbti: str | None,
    description: str | None,
    interests: Sequence[str],
    interview_topics: Sequence[str],
    interview_samples: Sequence[dict[str, Any]],
) -> ProfileQualityScore:
    normalized_answers = [
        _normalize_text(str(sample.get("transcript") or ""))
        for sample in interview_samples
    ]
    nonempty_answers = [answer for answer in normalized_answers if answer]
    placeholders = [
        answer for answer in nonempty_answers if _is_placeholder(answer)
    ]
    meaningful_answers = [
        answer
        for answer in nonempty_answers
        if not _is_placeholder(answer) and _information_length(answer) >= 4
    ]
    unique_meaningful = set(meaningful_answers)
    duplicate_count = max(len(meaningful_answers) - len(unique_meaningful), 0)

    profile_fields = (
        age,
        gender,
        mbti,
        description,
        list(interests),
    )
    profile_coverage = sum(_has_value(value) for value in profile_fields) / len(
        profile_fields
    )
    foundation_score = 20.0 * profile_coverage
    onboarding_score = 30.0 * min(len(unique_meaningful) / 5.0, 1.0)
    advanced_score = 30.0 * min(
        max(len(unique_meaningful) - 5, 0) / 10.0,
        1.0,
    )

    average_length = (
        sum(_information_length(answer) for answer in unique_meaningful)
        / len(unique_meaningful)
        if unique_meaningful
        else 0.0
    )
    depth_ratio = min(average_length / 80.0, 1.0)
    categories = {
        _normalize_text(str(sample.get("questionCategory") or ""))
        for sample in interview_samples
        if sample.get("questionCategory")
    }
    topic_sources = {
        *categories,
        *(
            _normalize_text(str(topic))
            for topic in interview_topics
            if _normalize_text(str(topic))
        ),
    }
    diversity_ratio = min(len(topic_sources) / 5.0, 1.0)
    depth_and_diversity_score = 20.0 * (
        0.60 * depth_ratio + 0.40 * diversity_ratio
    )
    profile_score = _score(
        foundation_score
        + onboarding_score
        + advanced_score
        + depth_and_diversity_score
    )

    total_count = max(len(nonempty_answers), 1)
    meaningful_ratio = len(meaningful_answers) / total_count
    unique_ratio = len(unique_meaningful) / max(len(meaningful_answers), 1)
    sourced_count = sum(
        bool(sample.get("questionText")) and bool(sample.get("transcript"))
        for sample in interview_samples
    )
    source_coverage = sourced_count / max(len(interview_samples), 1)
    data_reliability_score = (
        _score(
            40.0
            + 25.0 * meaningful_ratio
            + 15.0 * unique_ratio
            + 10.0 * source_coverage
            + 10.0 * profile_coverage
        )
        if nonempty_answers
        else _score(20.0 * profile_coverage)
    )

    penalty_score = _score(
        min(len(placeholders) * 2.0 + duplicate_count * 1.5, 20.0)
    )
    return ProfileQualityScore(
        profile_score=profile_score,
        data_reliability_score=data_reliability_score,
        penalty_score=penalty_score,
        meaningful_answer_count=len(unique_meaningful),
        duplicate_answer_count=duplicate_count,
        placeholder_answer_count=len(placeholders),
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _information_length(value: str) -> int:
    return len(re.sub(r"[^0-9a-z가-힣]", "", value, flags=re.IGNORECASE))


def _is_placeholder(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    return compact in _PLACEHOLDER_ANSWERS


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence):
        return any(_has_value(item) for item in value)
    return True


def _score(value: float) -> float:
    return round(max(0.0, min(float(value), 100.0)), 2)
