"""Member memory search used by the realtime call server.

The AI API server owns the only RAG store. When ``RAG_SEARCH_BASE_URL`` is
set (call server), memories are fetched from its internal search API instead
of opening a local ChromaDB, so every server reads the same store. Without a
base URL the local store is used (AI API server itself and local development).
"""

from __future__ import annotations

import threading
from typing import Any

import httpx

from model_training.rag_documents import PROFILE_SOURCE_TYPES
from shared.config import settings

INTERNAL_SEARCH_PATH = "/internal/rag/search"
INTERNAL_KEY_HEADER = "X-Rag-Internal-Key"

_client: httpx.Client | None = None
_client_lock = threading.Lock()


class RagSearchError(RuntimeError):
    code = "RAG_SEARCH_FAILED"


class RagSearchUnavailable(RagSearchError):
    code = "RAG_SEARCH_UNAVAILABLE"


class RagSearchConfigurationError(RagSearchError):
    code = "RAG_SEARCH_CONFIG_ERROR"


class RagSearchRejected(RagSearchError):
    code = "RAG_SEARCH_REJECTED"


def search_mode() -> str:
    return "remote" if settings.RAG_SEARCH_BASE_URL else "local"


def _http_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            # A pooled client keeps the connection to the AI server warm, so
            # each turn only pays for the search itself.
            _client = httpx.Client(
                timeout=httpx.Timeout(settings.RAG_SEARCH_TIMEOUT_SECONDS),
            )
        return _client


def _search_remote(user_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    api_key = settings.RAG_INTERNAL_API_KEY
    if not api_key:
        raise RagSearchConfigurationError(
            "RAG_INTERNAL_API_KEY is not configured."
        )
    timeout = settings.RAG_SEARCH_TIMEOUT_SECONDS
    if timeout <= 0:
        raise RagSearchConfigurationError(
            "RAG_SEARCH_TIMEOUT_SECONDS must be positive."
        )
    url = f"{settings.RAG_SEARCH_BASE_URL.rstrip('/')}{INTERNAL_SEARCH_PATH}"
    try:
        response = _http_client().post(
            url,
            headers={INTERNAL_KEY_HEADER: api_key},
            json={"userId": user_id, "query": query, "topK": top_k},
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise RagSearchUnavailable(
            f"RAG search API is unavailable ({type(exc).__name__})."
        ) from exc

    if response.status_code in {401, 403}:
        raise RagSearchConfigurationError(
            f"RAG search API rejected the internal key (HTTP {response.status_code})."
        )
    if response.status_code == 503:
        raise RagSearchConfigurationError(
            "RAG search API is not configured on the AI server (HTTP 503)."
        )
    if response.status_code >= 500:
        raise RagSearchUnavailable(
            f"RAG search API failed with HTTP {response.status_code}."
        )
    if not response.is_success:
        raise RagSearchRejected(
            f"RAG search API returned HTTP {response.status_code}."
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise RagSearchError("RAG search API returned invalid JSON.") from exc
    memories = body.get("memories") if isinstance(body, dict) else None
    if not isinstance(memories, list):
        raise RagSearchError("RAG search API response has no memories list.")
    return [memory for memory in memories if isinstance(memory, dict)]


def _memory_key(memory: dict[str, Any]) -> str:
    document_id = memory.get("documentId")
    if isinstance(document_id, str) and document_id:
        return f"id:{document_id}"
    return "text:" + " ".join(str(memory.get("text") or "").split()).casefold()


def _is_profile(memory: dict[str, Any]) -> bool:
    metadata = memory.get("metadata") or {}
    return str(metadata.get("sourceType", "")) in PROFILE_SOURCE_TYPES


def merge_memory_results(
    result_sets: list[list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge searches made with different queries for the same turn.

    The profile document is kept once at the front. Other memories are
    deduplicated (keeping the closest distance) and the ``top_k`` closest are
    returned, so a memory found only by one of the queries is not lost.
    """
    profiles: dict[str, dict[str, Any]] = {}
    memories: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for memory in results:
            key = _memory_key(memory)
            if _is_profile(memory):
                profiles.setdefault(key, memory)
                continue
            current = memories.get(key)
            if current is None or _distance(memory) < _distance(current):
                memories[key] = memory
    ranked = sorted(memories.values(), key=_distance)
    return [*profiles.values(), *ranked[: max(0, int(top_k))]]


def _distance(memory: dict[str, Any]) -> float:
    value = memory.get("distance")
    return float(value) if isinstance(value, (int, float)) else float("inf")


def search_user_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Blocking search; the realtime pipeline runs it in a worker thread."""
    if search_mode() == "remote":
        return _search_remote(user_id, query, top_k)

    # Local mode imports the store lazily so the call server never opens a
    # ChromaDB of its own when it is configured for remote search.
    from model_training.services import search_user_memories as local_search

    return local_search(user_id, query, top_k)
