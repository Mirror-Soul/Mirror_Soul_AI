"""Cost-free connection warm-up for the realtime call hot path."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

from model_calling.clients.rag_search import warmup_rag_connection
from model_calling.services import warmup_openai_connection
from shared.config import settings
from shared.elevenlabs_tts import warmup_elevenlabs_connection


async def _warmup_rag_async() -> str:
    return await asyncio.to_thread(warmup_rag_connection)


async def _run_one(
    name: str,
    operation: Awaitable[object],
    timeout_seconds: float,
) -> tuple[str, str, int]:
    started_at = time.monotonic()
    try:
        await asyncio.wait_for(operation, timeout=timeout_seconds)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return name, f"warning:{type(exc).__name__}", _elapsed_ms(started_at)
    return name, "ready", _elapsed_ms(started_at)


async def prewarm_realtime_dependencies(
    call_id: int,
) -> dict[str, str]:
    """Warm shared upstream connections before accepting a new call.

    Failures are advisory. The normal turn pipeline retains its own timeout
    and error handling and will retry the real operation when the user speaks.
    """
    if not settings.REALTIME_DEPENDENCY_PREWARM_ENABLED:
        return {"prewarm": "disabled"}
    timeout_seconds = settings.REALTIME_DEPENDENCY_PREWARM_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        return {"prewarm": "invalid-timeout"}

    started_at = time.monotonic()
    results = await asyncio.gather(
        _run_one("openai", warmup_openai_connection(), timeout_seconds),
        _run_one("rag", _warmup_rag_async(), timeout_seconds),
        _run_one(
            "elevenlabs",
            warmup_elevenlabs_connection(timeout_seconds=timeout_seconds),
            timeout_seconds,
        ),
    )
    statuses = {name: status for name, status, _ in results}
    timings = " ".join(
        f"{name}_ms={elapsed_ms}" for name, _, elapsed_ms in results
    )
    status_fields = " ".join(
        f"{name}={status}" for name, status, _ in results
    )
    print(
        "[REALTIME] dependencies prewarmed: "
        f"callId={call_id} {status_fields} {timings} "
        f"elapsed_ms={_elapsed_ms(started_at)}",
        flush=True,
    )
    return statuses


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))
