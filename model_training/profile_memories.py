from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_training.utils import (
    build_member_profile_summary_text,
    build_training_text,
    create_member_profile_document_id,
    create_member_profile_interview_document_id,
)


PROFILE_INTERVIEW_SOURCE_TYPE = "member_profile_interview"


@dataclass(frozen=True)
class ProfileMemoryDocument:
    document_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


def build_member_profile_documents(
    *,
    user_id: str,
    ai_profile_id: str | None,
    age: int | None,
    gender: str | None,
    mbti: str | None,
    description: str | None,
    keywords: list[str],
    interview_samples: list[dict[str, Any]],
) -> list[ProfileMemoryDocument]:
    profile_key = ai_profile_id or "default"
    profile_metadata: dict[str, str | int | float | bool] = {
        "userId": user_id,
        "sourceType": "member_profile_summary",
        "profileKey": profile_key,
        "keywordCount": len(keywords),
        "keywords": ", ".join(keywords),
    }
    if ai_profile_id:
        profile_metadata["aiProfileId"] = ai_profile_id
    if age is not None:
        profile_metadata["age"] = age
    if gender:
        profile_metadata["gender"] = gender
    if mbti:
        profile_metadata["mbti"] = mbti.upper()

    documents = [
        ProfileMemoryDocument(
            document_id=create_member_profile_document_id(user_id, ai_profile_id),
            text=build_member_profile_summary_text(
                age=age,
                gender=gender,
                mbti=mbti,
                keywords=keywords,
            ),
            metadata=profile_metadata,
        )
    ]

    for interview_index, sample in enumerate(interview_samples, start=1):
        transcript = str(sample.get("transcript") or "").strip()
        if not transcript:
            continue

        question_id = _optional_int(sample.get("questionId"))
        question_category = str(sample.get("questionCategory") or "").strip()
        question_text = str(sample.get("questionText") or "").strip()
        metadata: dict[str, str | int | float | bool] = {
            "userId": user_id,
            "sourceType": PROFILE_INTERVIEW_SOURCE_TYPE,
            "profileKey": profile_key,
            "interviewIndex": interview_index,
        }
        if ai_profile_id:
            metadata["aiProfileId"] = ai_profile_id
        if question_id is not None:
            metadata["questionId"] = question_id
        if question_category:
            metadata["questionCategory"] = question_category

        documents.append(
            ProfileMemoryDocument(
                document_id=create_member_profile_interview_document_id(
                    user_id,
                    ai_profile_id,
                    interview_index,
                    question_id,
                ),
                text=build_training_text(
                    mbti=mbti,
                    description=description,
                    question_category=question_category,
                    question_text=question_text,
                    transcript=transcript,
                ),
                metadata=metadata,
            )
        )

    return documents


def find_stale_profile_interview_ids(
    *,
    existing_ids: list[str],
    existing_metadatas: list[dict[str, Any] | None],
    user_id: str,
    ai_profile_id: str | None,
    current_ids: set[str],
) -> list[str]:
    profile_key = ai_profile_id or "default"
    stale_ids: list[str] = []
    for document_id, metadata in zip(existing_ids, existing_metadatas):
        metadata = metadata or {}
        if (
            metadata.get("userId") == user_id
            and metadata.get("sourceType") == PROFILE_INTERVIEW_SOURCE_TYPE
            and metadata.get("profileKey") == profile_key
            and document_id not in current_ids
        ):
            stale_ids.append(document_id)
    return stale_ids


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
