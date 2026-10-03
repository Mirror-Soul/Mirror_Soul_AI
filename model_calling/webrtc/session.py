import asyncio
from dataclasses import dataclass, field
from typing import Any

from aiortc import RTCPeerConnection

from model_calling.clients.backend_call_context import CallContext
from model_calling.realtime.trace import CallTrace


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

    def __post_init__(self) -> None:
        self.trace = CallTrace(
            call_id=self.call_id,
            user_id=self.clone_user_uuid,
            clone_id=self.clone_id,
            media_type=self.media_type,
        )


_sessions: dict[int, WebRTCSession] = {}
_call_contexts: dict[int, CallContext] = {}


def register_call_context(context: CallContext) -> None:
    _call_contexts[context.callId] = context


def get_call_context(call_id: int) -> CallContext | None:
    return _call_contexts.get(call_id)


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
            session.trace.close(reason=reason)
