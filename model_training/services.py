from typing import Any

import chromadb
from openai import OpenAI

from model_training.utils import (
    build_training_text,
    create_document_id,
    create_sample_id,
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
    response = openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding


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