from __future__ import annotations

from typing import Any, Mapping

from model_training.rag_documents import (
    INTERVIEW_MEMORY,
    RagDocument,
    build_interview_memory_document,
    build_profile_snapshot_document,
    find_stale_profile_document_ids,
    resolve_profile_key,
    utc_now_iso,
)

# Backwards compatible alias: profile interviews are now stored as the unified
# ``interview_memory`` source type.
PROFILE_INTERVIEW_SOURCE_TYPE = INTERVIEW_MEMORY

# Kept for callers/tests that still import the old dataclass name.
ProfileMemoryDocument = RagDocument


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
    clone_id: int | None = None,
    name: str | None = None,
    nickname: str | None = None,
    job: str | None = None,
    interests: list[str] | None = None,
    values: list[str] | None = None,
    updated_at: str | None = None,
) -> list[RagDocument]:
    """Build the profile snapshot plus one interview memory per answer.

    The first returned document is always the ``profile_snapshot``. Interview
    ids do not depend on the order of ``interview_samples``. When the same
    interview appears more than once in one request, the last one wins.
    """
    profile_key = resolve_profile_key(ai_profile_id)
    timestamp = updated_at or utc_now_iso()
    snapshot = build_profile_snapshot_document(
        user_id=user_id,
        profile_key=profile_key,
        clone_id=clone_id,
        ai_profile_id=ai_profile_id,
        name=name,
        nickname=nickname,
        age=age,
        gender=gender,
        mbti=mbti,
        job=job,
        description=description,
        interests=interests,
        values=values,
        keywords=keywords,
        updated_at=timestamp,
    )

    interviews: dict[str, RagDocument] = {}
    for sample in interview_samples:
        document = build_interview_memory_document(
            user_id=user_id,
            sample=sample,
            profile_key=profile_key,
            clone_id=clone_id,
            ai_profile_id=ai_profile_id,
            profile_managed=True,
            updated_at=timestamp,
        )
        if document is not None:
            interviews[document.document_id] = document

    return [snapshot, *interviews.values()]


def find_stale_profile_interview_ids(
    *,
    existing_ids: list[str],
    existing_metadatas: list[Mapping[str, Any] | None],
    user_id: str,
    ai_profile_id: str | None,
    current_ids: set[str],
) -> list[str]:
    return find_stale_profile_document_ids(
        existing_ids=existing_ids,
        existing_metadatas=existing_metadatas,
        user_id=user_id,
        profile_key=resolve_profile_key(ai_profile_id),
        current_ids=current_ids,
    )
