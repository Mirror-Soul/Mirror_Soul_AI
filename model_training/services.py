from typing import Any

import chromadb
from openai import OpenAI

from model_training.profile_memories import (
    PROFILE_INTERVIEW_SOURCE_TYPE,
    build_member_profile_documents,
    find_stale_profile_interview_ids,
)
from model_training.profile_quality import evaluate_profile_quality
from model_training.utils import (
    build_training_text,
    create_document_id,
    create_sample_id,
    extract_keywords_from_texts,
)
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


def add_training_sample_to_rag(
    user_id: str,
    ai_profile_id: str,
    question_id: int,
    question_category: str,
    question_text: str,
    transcript: str,
    mbti: str | None = None,
    description: str | None = None,
    audio_url: str | None = None,
) -> dict[str, Any]:
    sample_id = create_sample_id()
    document_id = create_document_id(sample_id)

    text = build_training_text(
        mbti=mbti,
        description=description,
        question_category=question_category,
        question_text=question_text,
        transcript=transcript,
    )

    embedding = create_embedding(text)

    metadata = {
        "userId": user_id,
        "aiProfileId": ai_profile_id,
        "sourceType": "interview_answer",
        "questionId": question_id,
        "questionCategory": question_category,
        "sampleId": sample_id,
    }

    if audio_url:
        metadata["audioUrl"] = audio_url

    collection.add(
        ids=[document_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )

    return {
        "documentId": document_id,
        "sampleId": sample_id,
        "status": "stored",
    }


def add_member_profile_to_rag(
    *,
    user_id: str,
    ai_profile_id: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    mbti: str | None = None,
    description: str | None = None,
    interests: list[str] | None = None,
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
        age=age,
        gender=gender,
        mbti=mbti,
        description=description,
        keywords=keywords,
        interview_samples=interview_samples or [],
    )
    embeddings = create_embeddings([document.text for document in documents])

    collection.upsert(
        ids=[document.document_id for document in documents],
        documents=[document.text for document in documents],
        embeddings=embeddings,
        metadatas=[document.metadata for document in documents],
    )

    current_interview_ids = {
        document.document_id
        for document in documents
        if document.metadata.get("sourceType") == PROFILE_INTERVIEW_SOURCE_TYPE
    }
    existing = collection.get(
        where={"userId": user_id},
        include=["metadatas"],
    )
    stale_ids = find_stale_profile_interview_ids(
        existing_ids=existing.get("ids") or [],
        existing_metadatas=existing.get("metadatas") or [],
        user_id=user_id,
        ai_profile_id=ai_profile_id,
        current_ids=current_interview_ids,
    )
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "documentId": documents[0].document_id,
        "status": "stored",
        "keywords": keywords,
        "profileSummary": documents[0].text,
        "profileQuality": profile_quality.to_dict(),
    }


def search_user_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"userId": user_id},
    )

    memories: list[dict[str, Any]] = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, document_id in enumerate(ids):
        memories.append(
            {
                "documentId": document_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return memories


def delete_user_rag_data(user_id: str) -> dict[str, Any]:
    collection.delete(where={"userId": user_id})

    return {
        "success": True,
        "userId": user_id,
        "deleted": "rag_documents",
    }
