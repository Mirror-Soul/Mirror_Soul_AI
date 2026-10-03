from typing import Any

import chromadb
from openai import OpenAI

from model_training.profile_memories import build_member_profile_documents
from model_training.profile_quality import evaluate_profile_quality
from model_training.rag_documents import (
    PROFILE_SNAPSHOT,
    PROFILE_SOURCE_TYPES,
    RagDocument,
    build_interview_memory_document,
    find_stale_profile_document_ids,
    memory_dedupe_key,
    resolve_profile_key,
    schema_version,
)
from model_training.utils import extract_keywords_from_texts
from shared.config import settings

if not settings.OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please check your .env file.")

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=settings.RAG_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=settings.RAG_COLLECTION_NAME
)


def create_embedding(text: str) -> list[float]:
    return create_embeddings([text])[0]


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    if len(ordered) != len(texts):
        raise RuntimeError("Embedding response count does not match input count")
    return [item.embedding for item in ordered]


def upsert_rag_documents(documents: list[RagDocument]) -> list[str]:
    """Single write path for every RAG document.

    Embeddings are created in one batch and documents are written with
    deterministic ids via ``upsert`` so retries never add duplicates.
    """
    if not documents:
        return []
    unique: dict[str, RagDocument] = {}
    for document in documents:
        unique[document.document_id] = document
    ordered = list(unique.values())
    embeddings = create_embeddings([document.text for document in ordered])
    collection.upsert(
        ids=[document.document_id for document in ordered],
        documents=[document.text for document in ordered],
        embeddings=embeddings,
        metadatas=[document.metadata for document in ordered],
    )
    return [document.document_id for document in ordered]


def _existing_metadata(document_id: str) -> dict[str, Any] | None:
    stored = collection.get(ids=[document_id], include=["metadatas"])
    metadatas = stored.get("metadatas") or []
    if not metadatas:
        return None
    return dict(metadatas[0] or {})


def add_training_sample_to_rag(
    user_id: str,
    ai_profile_id: str | None,
    question_id: int | None,
    question_category: str | None,
    question_text: str | None,
    transcript: str,
    mbti: str | None = None,
    description: str | None = None,
    audio_url: str | None = None,
    source_id: str | None = None,
    clone_id: int | None = None,
) -> dict[str, Any]:
    """Store one interview answer through the shared upsert path.

    ``mbti`` and ``description`` are accepted for request compatibility but
    belong to the profile snapshot, so they are not repeated in every
    interview memory. If the same interview was already stored by the profile
    sync, its ownership (``profileManaged``/``profileKey``) is preserved.
    """
    del mbti, description
    sample = {
        "sourceId": source_id,
        "questionId": question_id,
        "questionCategory": question_category,
        "questionText": question_text,
        "transcript": transcript,
    }
    profile_key = resolve_profile_key(ai_profile_id)
    probe = build_interview_memory_document(
        user_id=user_id,
        sample=sample,
        profile_key=profile_key,
        profile_managed=False,
    )
    if probe is None:
        raise ValueError("transcript must not be blank")

    existing = _existing_metadata(probe.document_id) or {}
    profile_managed = existing.get("profileManaged") is True
    if profile_managed:
        profile_key = str(existing.get("profileKey") or profile_key)
    if clone_id is None:
        existing_clone_id = existing.get("cloneId")
        clone_id = existing_clone_id if isinstance(existing_clone_id, int) else None

    document = build_interview_memory_document(
        user_id=user_id,
        sample=sample,
        profile_key=profile_key,
        clone_id=clone_id,
        ai_profile_id=ai_profile_id,
        profile_managed=profile_managed,
        extra={"audioUrl": audio_url},
    )
    assert document is not None
    upsert_rag_documents([document])

    return {
        "documentId": document.document_id,
        "sampleId": document.source_id,
        "status": "stored",
    }


def add_member_profile_to_rag(
    *,
    user_id: str,
    ai_profile_id: str | None = None,
    clone_id: int | None = None,
    name: str | None = None,
    nickname: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    mbti: str | None = None,
    job: str | None = None,
    description: str | None = None,
    interests: list[str] | None = None,
    values: list[str] | None = None,
    interview_topics: list[str] | None = None,
    interview_samples: list[dict[str, Any]] | None = None,
    keyword_limit: int = 12,
) -> dict[str, Any]:
    profile_quality = evaluate_profile_quality(
        age=age,
        gender=gender,
        mbti=mbti,
        description=description,
        interests=interests or [],
        interview_topics=interview_topics or [],
        interview_samples=interview_samples or [],
    )
    seed_keywords = [
        *(interests or []),
        *(values or []),
        *(interview_topics or []),
    ]

    keyword_source_texts: list[str] = []
    if description:
        keyword_source_texts.append(description)

    for sample in interview_samples or []:
        question_category = sample.get("questionCategory")
        question_text = sample.get("questionText")
        transcript = sample.get("transcript")

        if question_category:
            seed_keywords.append(str(question_category))
        if question_text:
            keyword_source_texts.append(str(question_text))
        if transcript:
            keyword_source_texts.append(str(transcript))

    keywords = extract_keywords_from_texts(
        keyword_source_texts,
        seed_keywords=seed_keywords,
        limit=keyword_limit,
    )

    documents = build_member_profile_documents(
        user_id=user_id,
        ai_profile_id=ai_profile_id,
        clone_id=clone_id,
        name=name,
        nickname=nickname,
        age=age,
        gender=gender,
        mbti=mbti,
        job=job,
        description=description,
        interests=interests or [],
        values=values or [],
        keywords=keywords,
        interview_samples=interview_samples or [],
    )
    current_ids = set(upsert_rag_documents(documents))

    # Only after the new documents are stored, remove documents that this
    # profile sync owned before but which are no longer part of the request.
    existing = collection.get(
        where={"userId": user_id},
        include=["metadatas"],
    )
    stale_ids = find_stale_profile_document_ids(
        existing_ids=existing.get("ids") or [],
        existing_metadatas=existing.get("metadatas") or [],
        user_id=user_id,
        profile_key=resolve_profile_key(ai_profile_id),
        current_ids=current_ids,
    )
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "documentId": documents[0].document_id,
        "status": "stored",
        "keywords": keywords,
        "profileSummary": documents[0].text,
        "profileQuality": profile_quality.to_dict(),
        "documentCount": len(current_ids),
        "removedDocumentCount": len(stale_ids),
    }


def _get_user_profile_document(user_id: str) -> dict[str, Any] | None:
    """Exact lookup of the profile document, independent of vector similarity.

    A v2 ``profile_snapshot`` wins over a legacy ``member_profile_summary``;
    among several candidates the most recently updated one is used.
    """
    stored = collection.get(
        where={
            "$and": [
                {"userId": user_id},
                {"sourceType": {"$in": sorted(PROFILE_SOURCE_TYPES)}},
            ]
        },
        include=["documents", "metadatas"],
    )
    candidates: list[dict[str, Any]] = []
    for document_id, document, metadata in zip(
        stored.get("ids") or [],
        stored.get("documents") or [],
        stored.get("metadatas") or [],
    ):
        metadata = dict(metadata or {})
        if metadata.get("userId") != user_id:
            continue
        candidates.append(
            {
                "documentId": document_id,
                "text": document,
                "metadata": metadata,
                "distance": None,
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["metadata"].get("sourceType") == PROFILE_SNAPSHOT,
            schema_version(item["metadata"]),
            str(item["metadata"].get("updatedAt") or ""),
            item["documentId"],
        ),
    )


def _search_memory_documents(
    user_id: str,
    query_embedding: list[float],
    *,
    top_k: int,
    distance_limit: float,
) -> list[dict[str, Any]]:
    fetch_count = min(max(top_k * 3, top_k + 4), 50)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_count,
        where={
            "$and": [
                {"userId": user_id},
                {"sourceType": {"$nin": sorted(PROFILE_SOURCE_TYPES)}},
            ]
        },
    )

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    best_by_key: dict[str, dict[str, Any]] = {}
    for index, document_id in enumerate(ids):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        if metadata.get("userId") != user_id:
            continue
        if metadata.get("sourceType") in PROFILE_SOURCE_TYPES:
            continue
        distance = distances[index] if index < len(distances) else None
        if distance is not None and distance > distance_limit:
            continue
        text = documents[index] if index < len(documents) else ""
        memory = {
            "documentId": document_id,
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }
        key = memory_dedupe_key(text, metadata)
        current = best_by_key.get(key)
        if current is None or _memory_rank(memory) < _memory_rank(current):
            best_by_key[key] = memory

    ordered = sorted(best_by_key.values(), key=_distance_sort_key)
    return ordered[:top_k]


def _distance_sort_key(memory: dict[str, Any]) -> tuple[float, str]:
    distance = memory.get("distance")
    return (float("inf") if distance is None else float(distance), memory["documentId"])


def _memory_rank(memory: dict[str, Any]) -> tuple[int, float, str]:
    # Prefer the newest schema, then the closest document.
    return (-schema_version(memory["metadata"]), *_distance_sort_key(memory))


def search_user_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Return the profile document (once) followed by relevant memories.

    The profile is fetched exactly and does not occupy a vector search slot,
    so up to ``top_k`` memories are returned in addition to the profile.
    All lookups are restricted to ``user_id``.
    """
    if not str(user_id or "").strip():
        return []
    distance_limit = (
        settings.RAG_MAX_DISTANCE if max_distance is None else max_distance
    )
    top_k = max(0, int(top_k))

    profile = _get_user_profile_document(user_id)
    memories: list[dict[str, Any]] = []
    if top_k > 0 and str(query or "").strip():
        memories = _search_memory_documents(
            user_id,
            create_embedding(query),
            top_k=top_k,
            distance_limit=distance_limit,
        )
    if profile is not None:
        return [profile, *memories]
    return memories


def delete_user_rag_data(user_id: str) -> dict[str, Any]:
    collection.delete(where={"userId": user_id})

    return {
        "success": True,
        "userId": user_id,
        "deleted": "rag_documents",
    }
