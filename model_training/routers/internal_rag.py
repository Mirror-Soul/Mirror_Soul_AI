"""Internal member-memory search API used by the realtime call server.

The AI API server owns the only RAG store, so other AI components (the call
server) search it through this endpoint instead of opening their own ChromaDB.
Requests must carry the shared ``X-Rag-Internal-Key`` header.
"""

from __future__ import annotations

import hmac
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from model_training.schemas import MemoryItem
from shared.config import settings

router = APIRouter(prefix="/internal/rag", tags=["internal_rag"])


class InternalRagSearchRequest(BaseModel):
    userId: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    topK: int = Field(6, ge=1, le=20)


class InternalRagSearchResponse(BaseModel):
    success: bool
    memories: list[MemoryItem]


def _require_internal_key(provided: str | None) -> None:
    expected = settings.RAG_INTERNAL_API_KEY
    if not expected:
        # Never serve member memories without authentication.
        raise HTTPException(status_code=503, detail="RAG internal API is not configured.")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid internal key.")


@router.post("/search", response_model=InternalRagSearchResponse)
def internal_search(
    request: InternalRagSearchRequest,
    x_rag_internal_key: str | None = Header(default=None),
):
    _require_internal_key(x_rag_internal_key)

    # Imported lazily so this router module stays cheap to import.
    from model_training.services import search_user_memories

    started = time.monotonic()
    memories = search_user_memories(
        user_id=request.userId,
        query=request.query,
        top_k=request.topK,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(
        "[RAG_SEARCH] served: "
        f"user_uuid={request.userId} count={len(memories)} "
        f"query_chars={len(request.query)} elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return InternalRagSearchResponse(success=True, memories=memories)
