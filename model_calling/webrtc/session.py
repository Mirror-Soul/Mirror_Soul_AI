import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aiortc import RTCPeerConnection

from model_calling.clients.backend_call_context import CallContext
from model_calling.realtime.talk_log import CallTalkLogRecorder
from model_calling.realtime.trace import CallTrace
from shared.config import settings


@dataclass
class WebRTCSession:
    call_id: int
    room_id: str
    ai_signal_id: str
    caller_signal_id: str
    peer_connection: RTCPeerConnection
    clone_user_uuid: str
    clone_id: int
    output_track: Any
    utterance_queue: asyncio.Queue[Any]
    media_type: str = "VOICE"
    output_video_track: Any | None = None
    video_renderer: Any | None = None
    video_prepare_task: asyncio.Task | None = None
    pipeline_start_task: asyncio.Task | None = None
    receiver_task: asyncio.Task | None = None
    pipeline_task: asyncio.Task | None = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    trace: CallTrace = field(init=False)
    talk_log: CallTalkLogRecorder = field(init=False)

    def __post_init__(self) -> None:
        self.trace = CallTrace(
            call_id=self.call_id,
            user_id=self.clone_user_uuid,
            clone_id=self.clone_id,
            media_type=self.media_type,
        )
        self.talk_log = CallTalkLogRecorder(self.call_id)


_sessions: dict[int, WebRTCSession] = {}
# callId -> (context, registered monotonic time). The backend context is
# fetched once on CALL_INVITE and reused for every turn of that call.
_call_contexts: dict[int, tuple[CallContext, float]] = {}


def _now() -> float:
    return time.monotonic()


def _is_expired(registered_at: float, now: float) -> bool:
    ttl = settings.CALL_CONTEXT_CACHE_TTL_SECONDS
    return ttl > 0 and now - registered_at > ttl


def prune_call_contexts(now: float | None = None) -> int:
    """Drop contexts of calls that never reached (or already left) a session.

    Contexts that belong to an active WebRTC session are never evicted, so a
    long call keeps its context until CALL_END / peer close. Returns the
    number of removed entries.
    """
    current = _now() if now is None else now
    removed = 0
    for call_id, (_, registered_at) in list(_call_contexts.items()):
        if call_id not in _sessions and _is_expired(registered_at, current):
            _call_contexts.pop(call_id, None)
            removed += 1

    max_entries = max(1, int(settings.CALL_CONTEXT_CACHE_MAX_ENTRIES))
    if len(_call_contexts) > max_entries:
        idle = sorted(
            (
                (registered_at, call_id)
                for call_id, (_, registered_at) in _call_contexts.items()
                if call_id not in _sessions
            ),
        )
        for _, call_id in idle[: len(_call_contexts) - max_entries]:
            _call_contexts.pop(call_id, None)
            removed += 1
    return removed


def register_call_context(context: CallContext) -> None:
    _call_contexts[context.callId] = (context, _now())
    prune_call_contexts()


def get_call_context(call_id: int) -> CallContext | None:
    entry = _call_contexts.get(call_id)
    if entry is None:
        return None
    context, registered_at = entry
    if call_id not in _sessions and _is_expired(registered_at, _now()):
        _call_contexts.pop(call_id, None)
        return None
    return context


def call_context_count() -> int:
    return len(_call_contexts)


def get_call_user(call_id: int) -> str | None:
    context = get_call_context(call_id)
    return context.clone.userUuid if context else None


def get_call_clone_id(call_id: int) -> int | None:
    context = get_call_context(call_id)
    return context.clone.cloneId if context else None


def get_call_media_type(call_id: int) -> str | None:
    context = get_call_context(call_id)
    return context.mediaType if context else None


def save_session(session: WebRTCSession) -> None:
    _sessions[session.call_id] = session


def get_session(call_id: int) -> WebRTCSession | None:
    return _sessions.get(call_id)


async def close_session(call_id: int, *, reason: str = "SESSION_CLOSED") -> None:
    _call_contexts.pop(call_id, None)
    session = _sessions.pop(call_id, None)
    if session:
        cancelled_tasks: list[asyncio.Task] = []
        current_task = asyncio.current_task()
        for task in (
            session.receiver_task,
            session.pipeline_task,
            session.video_prepare_task,
            session.pipeline_start_task,
        ):
            if task and task is not current_task and not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            print(
                "[CALL_TRACE] in-flight tasks cancelled: "
                f"callId={call_id} count={len(cancelled_tasks)}",
                flush=True,
            )
        try:
            await session.peer_connection.close()
        finally:
            try:
                # Utterances already spoken are still saved to the call
                # history, bounded by BACKEND_TALK_LOG_FLUSH_TIMEOUT_SECONDS.
                await session.talk_log.close()
            finally:
                session.trace.close(reason=reason)
