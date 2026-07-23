import os
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CloneSimilaritySnapshot:
    clone_id: int
    user_uuid: str
    name: str | None
    gender: str | None
    birth_date: date | None
    job: str | None
    job_description: str | None
    self_introduction: str | None
    mbti: str | None
    voice_profile_id: int | None
    voice_training_job_id: int | None
    elevenlabs_voice_id: str | None
    voice_training_status: str | None
    voice_training_audio_count: int
    interview_answer_count: int
    interview_text_count: int
    interview_audio_count: int


@dataclass(frozen=True)
class CloneSimilarityScore:
    clone_id: int
    voice_profile_id: int | None
    voice_training_job_id: int | None
    total_score: float
    voice_score: float
    interview_score: float
    profile_score: float
    explanation: str


def calculate_clone_similarity(
    snapshot: CloneSimilaritySnapshot,
    *,
    actual_voice_score: float | None = None,
) -> CloneSimilarityScore:
    voice_score = (
        _round_score(actual_voice_score)
        if actual_voice_score is not None
        else _calculate_voice_score(snapshot)
    )
    interview_score = _calculate_interview_score(snapshot)
    profile_score = _calculate_profile_score(snapshot)

    voice_weight = _env_float("CLONE_SIMILARITY_VOICE_WEIGHT", 0.60)
    interview_weight = _env_float("CLONE_SIMILARITY_INTERVIEW_WEIGHT", 0.25)
    profile_weight = _env_float("CLONE_SIMILARITY_PROFILE_WEIGHT", 0.15)
    weight_sum = voice_weight + interview_weight + profile_weight
    if weight_sum <= 0:
        voice_weight, interview_weight, profile_weight, weight_sum = 0.60, 0.25, 0.15, 1.0

    total_score = (
        voice_score * voice_weight
        + interview_score * interview_weight
        + profile_score * profile_weight
    ) / weight_sum

    return CloneSimilarityScore(
        clone_id=snapshot.clone_id,
        voice_profile_id=snapshot.voice_profile_id,
        voice_training_job_id=snapshot.voice_training_job_id,
        total_score=_round_score(total_score),
        voice_score=_round_score(voice_score),
        interview_score=_round_score(interview_score),
        profile_score=_round_score(profile_score),
        explanation=_build_explanation(
            voice_score=voice_score,
            interview_score=interview_score,
            profile_score=profile_score,
            snapshot=snapshot,
            used_speaker_embedding=actual_voice_score is not None,
        ),
    )


def _calculate_voice_score(snapshot: CloneSimilaritySnapshot) -> float:
    if not snapshot.elevenlabs_voice_id or snapshot.voice_training_status != "COMPLETED":
        return 0.0

    expected_samples = _env_int("CLONE_SIMILARITY_EXPECTED_VOICE_SAMPLES", 5)
    excellent_samples = _env_int("CLONE_SIMILARITY_EXCELLENT_VOICE_SAMPLES", 20)
    sample_count = snapshot.voice_training_audio_count

    # This is a readiness-based proxy. Replace the sample component with
    # speaker-embedding cosine similarity when the voice evaluator is added.
    if sample_count <= 0:
        return 0.0
    if sample_count <= expected_samples:
        sample_ratio = sample_count / max(expected_samples, 1)
        return 30.0 + 32.0 * sample_ratio

    extra_ratio = (sample_count - expected_samples) / max(
        excellent_samples - expected_samples,
        1,
    )
    return 62.0 + 33.0 * min(extra_ratio, 1.0)


def _calculate_interview_score(snapshot: CloneSimilaritySnapshot) -> float:
    expected_interviews = _env_int("CLONE_SIMILARITY_EXPECTED_INTERVIEWS", 5)
    excellent_interviews = _env_int("CLONE_SIMILARITY_EXCELLENT_INTERVIEWS", 15)
    answer_ratio = min(snapshot.interview_answer_count / max(expected_interviews, 1), 1.0)
    text_ratio = min(snapshot.interview_text_count / max(expected_interviews, 1), 1.0)
    audio_ratio = min(snapshot.interview_audio_count / max(expected_interviews, 1), 1.0)

    base_coverage = answer_ratio * 0.5 + text_ratio * 0.25 + audio_ratio * 0.25
    base_score = 25.0 + 37.0 * base_coverage

    extra_ratio = max(snapshot.interview_answer_count - expected_interviews, 0) / max(
        excellent_interviews - expected_interviews,
        1,
    )
    return base_score + 33.0 * min(extra_ratio, 1.0)


def _calculate_profile_score(snapshot: CloneSimilaritySnapshot) -> float:
    fields = [
        snapshot.name,
        snapshot.gender,
        snapshot.birth_date,
        snapshot.job,
        snapshot.self_introduction,
        snapshot.mbti,
    ]
    completed = sum(1 for value in fields if _has_value(value))
    core_score = 20.0 + 45.0 * completed / len(fields)
    detail_bonus = 5.0 if _has_value(snapshot.job_description) else 0.0
    return core_score + detail_bonus


def _build_explanation(
    *,
    voice_score: float,
    interview_score: float,
    profile_score: float,
    snapshot: CloneSimilaritySnapshot,
    used_speaker_embedding: bool,
) -> str:
    voice_basis = (
        "원본 녹음과 클론 음성의 스피커 임베딩 유사도"
        if used_speaker_embedding
        else "생성된 목소리 클론과 음성 샘플 수"
    )
    return (
        f"{voice_basis}, 온보딩 인터뷰 답변, "
        "기본 프로필 정보를 함께 반영해 계산한 현재 클론 유사도입니다. "
        f"음성 샘플 {snapshot.voice_training_audio_count}개, "
        f"인터뷰 답변 {snapshot.interview_answer_count}개, "
        "프로필 완성도를 기준으로 업데이트되었습니다."
    )


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _round_score(score: float) -> float:
    return round(max(0.0, min(score, 100.0)), 2)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default
