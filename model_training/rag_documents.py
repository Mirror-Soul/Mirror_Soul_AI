"""Unified RAG document model shared by every RAG write and read path.

This module is intentionally free of side effects (no Chroma or OpenAI
clients) so that both the training API and the realtime calling code can
import the shared constants and helpers.

Document id rule::

    {userId}:{sourceType}:{sourceId}

The same source always maps to the same id, so writes are ``upsert`` based
and retries never create duplicate documents.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

RAG_SCHEMA_VERSION = 2

# Current (schema v2) source types.
PROFILE_SNAPSHOT = "profile_snapshot"
INTERVIEW_MEMORY = "interview_memory"
CONVERSATION_MEMORY = "conversation_memory"  # reserved for future use
PREFERENCE_MEMORY = "preference_memory"  # reserved for future use

# Legacy (schema v1) source types that must stay readable during migration.
LEGACY_PROFILE_SUMMARY = "member_profile_summary"
LEGACY_PROFILE_INTERVIEW = "member_profile_interview"
LEGACY_INTERVIEW_ANSWER = "interview_answer"

PROFILE_SOURCE_TYPES = frozenset({PROFILE_SNAPSHOT, LEGACY_PROFILE_SUMMARY})
INTERVIEW_SOURCE_TYPES = frozenset(
    {INTERVIEW_MEMORY, LEGACY_PROFILE_INTERVIEW, LEGACY_INTERVIEW_ANSWER}
)
MEMORY_SOURCE_TYPES = frozenset(
    {*INTERVIEW_SOURCE_TYPES, CONVERSATION_MEMORY, PREFERENCE_MEMORY}
)
LEGACY_PROFILE_MANAGED_SOURCE_TYPES = frozenset(
    {LEGACY_PROFILE_SUMMARY, LEGACY_PROFILE_INTERVIEW}
)

DEFAULT_PROFILE_KEY = "default"
PROFILE_SNAPSHOT_IMPORTANCE = 1.0
INTERVIEW_MEMORY_IMPORTANCE = 0.8
USER_PROVIDED_CONFIDENCE = 1.0
DESCRIPTION_MAX_CHARS = 500

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class RagDocument:
    """A single searchable RAG document following the v2 schema."""

    user_id: str
    source_type: str
    source_id: str
    text: str
    profile_key: str = DEFAULT_PROFILE_KEY
    clone_id: int | None = None
    importance: float = INTERVIEW_MEMORY_IMPORTANCE
    confidence: float = USER_PROVIDED_CONFIDENCE
    profile_managed: bool = False
    updated_at: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return build_document_id(self.user_id, self.source_type, self.source_id)

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            "userId": self.user_id,
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "profileKey": self.profile_key,
            "schemaVersion": RAG_SCHEMA_VERSION,
            "updatedAt": self.updated_at or utc_now_iso(),
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "profileManaged": bool(self.profile_managed),
        }
        if self.clone_id is not None:
            metadata["cloneId"] = int(self.clone_id)
        metadata.update(sanitize_metadata(self.extra))
        return metadata


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_document_id(user_id: str, source_type: str, source_id: str) -> str:
    user_id = str(user_id).strip()
    source_type = str(source_type).strip()
    source_id = str(source_id).strip()
    if not user_id or not source_type or not source_id:
        raise ValueError("userId, sourceType and sourceId are required")
    return f"{user_id}:{source_type}:{source_id}"


def resolve_profile_key(ai_profile_id: str | None) -> str:
    """Profile key compatible with schema v1 documents (aiProfileId or default)."""
    value = str(ai_profile_id or "").strip()
    return value or DEFAULT_PROFILE_KEY


def sanitize_metadata(values: Mapping[str, Any]) -> dict[str, MetadataValue]:
    """Keep only Chroma compatible scalar metadata values (drop None/lists)."""
    sanitized: dict[str, MetadataValue] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, str) and not value.strip():
                continue
            sanitized[str(key)] = value
    return sanitized


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_interview_source_id(
    *,
    source_id: Any = None,
    question_id: Any = None,
    question_text: Any = None,
    transcript: Any = None,
) -> str:
    """Return an order independent, stable source id for an interview answer.

    Priority: explicit backend ``sourceId`` > ``questionId`` > hash of the
    normalized question text > hash of the normalized answer (legacy requests
    without any identifier).
    """
    explicit = compact_text(source_id)
    if explicit:
        return explicit
    parsed_question_id = optional_int(question_id)
    if parsed_question_id is not None:
        return f"question-{parsed_question_id}"
    normalized_question = compact_text(question_text).casefold()
    if normalized_question:
        return f"question-text-{_stable_hash(normalized_question)}"
    normalized_answer = compact_text(transcript).casefold()
    if normalized_answer:
        return f"answer-text-{_stable_hash(normalized_answer)}"
    raise ValueError("interview memory requires a sourceId, questionId or text")


def build_interview_memory_text(
    *,
    question_category: str | None,
    question_text: str | None,
    transcript: str,
) -> str:
    """Searchable text for one interview answer (profile data lives elsewhere)."""
    sections = ["[인터뷰 질문]"]
    category = compact_text(question_category)
    question = compact_text(question_text)
    if category:
        sections.append(f"카테고리: {category}")
    if question:
        sections.append(f"질문: {question}")
    sections.extend(["", "[사용자 답변]", str(transcript).strip()])
    return "\n".join(sections)


def build_interview_memory_document(
    *,
    user_id: str,
    sample: Mapping[str, Any],
    profile_key: str,
    clone_id: int | None = None,
    ai_profile_id: str | None = None,
    profile_managed: bool,
    updated_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RagDocument | None:
    transcript = str(sample.get("transcript") or "").strip()
    if not transcript:
        return None

    question_id = optional_int(sample.get("questionId"))
    question_category = compact_text(sample.get("questionCategory"))
    question_text = compact_text(sample.get("questionText"))
    source_id = resolve_interview_source_id(
        source_id=sample.get("sourceId"),
        question_id=question_id,
        question_text=question_text,
        transcript=transcript,
    )
    metadata_extra: dict[str, Any] = {
        "questionId": question_id,
        "questionCategory": question_category,
        "aiProfileId": ai_profile_id,
    }
    metadata_extra.update(extra or {})

    return RagDocument(
        user_id=user_id,
        source_type=INTERVIEW_MEMORY,
        source_id=source_id,
        text=build_interview_memory_text(
            question_category=question_category,
            question_text=question_text,
            transcript=transcript,
        ),
        profile_key=profile_key,
        clone_id=clone_id,
        importance=INTERVIEW_MEMORY_IMPORTANCE,
        confidence=USER_PROVIDED_CONFIDENCE,
        profile_managed=profile_managed,
        updated_at=updated_at or utc_now_iso(),
        extra=metadata_extra,
    )


def _clean_list(values: Iterable[Any] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = compact_text(value)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            cleaned.append(text)
    return cleaned


def build_profile_snapshot_text(
    *,
    name: str | None = None,
    nickname: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    mbti: str | None = None,
    job: str | None = None,
    description: str | None = None,
    interests: Iterable[str] | None = None,
    values: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
) -> str:
    """Build the profile snapshot using only values confirmed by the backend.

    Missing fields are omitted instead of being guessed or filled with
    placeholders.
    """
    lines = ["[회원 핵심 프로필]"]
    if compact_text(name):
        lines.append(f"이름: {compact_text(name)}")
    if compact_text(nickname):
        lines.append(f"닉네임: {compact_text(nickname)}")
    if age is not None:
        lines.append(f"나이: {age}")
    if compact_text(gender):
        lines.append(f"성별: {compact_text(gender)}")
    if compact_text(mbti):
        lines.append(f"MBTI: {compact_text(mbti).upper()}")
    if compact_text(job):
        lines.append(f"직업: {compact_text(job)}")
    intro = compact_text(description)
    if intro:
        if len(intro) > DESCRIPTION_MAX_CHARS:
            intro = f"{intro[: DESCRIPTION_MAX_CHARS - 1].rstrip()}…"
        lines.append(f"자기소개: {intro}")

    for heading, items in (
        ("[관심사]", _clean_list(interests)),
        ("[가치관]", _clean_list(values)),
        ("[핵심 키워드]", _clean_list(keywords)),
    ):
        if items:
            lines.extend(["", heading, *(f"- {item}" for item in items)])

    return "\n".join(lines)


def build_profile_snapshot_document(
    *,
    user_id: str,
    profile_key: str,
    clone_id: int | None = None,
    ai_profile_id: str | None = None,
    name: str | None = None,
    nickname: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    mbti: str | None = None,
    job: str | None = None,
    description: str | None = None,
    interests: Iterable[str] | None = None,
    values: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    updated_at: str | None = None,
) -> RagDocument:
    keyword_list = _clean_list(keywords)
    return RagDocument(
        user_id=user_id,
        source_type=PROFILE_SNAPSHOT,
        source_id=profile_key,
        text=build_profile_snapshot_text(
            name=name,
            nickname=nickname,
            age=age,
            gender=gender,
            mbti=mbti,
            job=job,
            description=description,
            interests=interests,
            values=values,
            keywords=keyword_list,
        ),
        profile_key=profile_key,
        clone_id=clone_id,
        importance=PROFILE_SNAPSHOT_IMPORTANCE,
        confidence=USER_PROVIDED_CONFIDENCE,
        profile_managed=True,
        updated_at=updated_at or utc_now_iso(),
        extra={
            "aiProfileId": ai_profile_id,
            "age": age,
            "gender": compact_text(gender),
            "mbti": compact_text(mbti).upper(),
            "keywordCount": len(keyword_list),
            "keywords": ", ".join(keyword_list),
        },
    )


def is_profile_source(metadata: Mapping[str, Any] | None) -> bool:
    return str((metadata or {}).get("sourceType", "")) in PROFILE_SOURCE_TYPES


def find_stale_profile_document_ids(
    *,
    existing_ids: list[str],
    existing_metadatas: list[Mapping[str, Any] | None],
    user_id: str,
    profile_key: str,
    current_ids: set[str],
) -> list[str]:
    """Documents owned by this profile sync that are no longer current.

    Only documents of the same user and profileKey that were written by the
    profile sync (v2 ``profileManaged`` documents or legacy v1 profile
    documents) are returned. Independent memories, other profile keys and
    other users are never selected.
    """
    stale: list[str] = []
    for document_id, metadata in zip(existing_ids, existing_metadatas):
        metadata = metadata or {}
        if document_id in current_ids:
            continue
        if metadata.get("userId") != user_id:
            continue
        if metadata.get("profileKey") != profile_key:
            continue
        source_type = metadata.get("sourceType")
        if source_type in LEGACY_PROFILE_MANAGED_SOURCE_TYPES:
            stale.append(document_id)
        elif source_type in {PROFILE_SNAPSHOT, INTERVIEW_MEMORY} and (
            metadata.get("profileManaged") is True
        ):
            stale.append(document_id)
    return stale


def memory_dedupe_key(text: Any, metadata: Mapping[str, Any] | None) -> str:
    """Key used to collapse the same memory stored in v1 and v2 formats."""
    metadata = metadata or {}
    if metadata.get("sourceType") in INTERVIEW_SOURCE_TYPES:
        question_id = optional_int(metadata.get("questionId"))
        if question_id is not None:
            return f"interview-question:{question_id}"
        source_id = compact_text(metadata.get("sourceId"))
        if source_id:
            return f"interview-source:{source_id}"
    return f"text:{compact_text(text).casefold()}"


def schema_version(metadata: Mapping[str, Any] | None) -> int:
    return optional_int((metadata or {}).get("schemaVersion")) or 1
