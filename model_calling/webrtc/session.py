import asyncio
from dataclasses import dataclass
from typing import Any

from aiortc import RTCPeerConnection


@dataclass
class WebRTCSession:
    call_id: int
    room_id: str
    ai_signal_id: str
    caller_signal_id: str
    peer_connection: RTCPeerConnection
    clone_user_uuid: str
    output_track: Any
    utterance_queue: asyncio.Queue[bytes]
    receiver_task: asyncio.Task | None = None
    pipeline_task: asyncio.Task | None = None


_sessions: dict[int, WebRTCSession] = {}
_call_users: dict[int, str] = {}


def register_call_user(call_id: int, clone_user_uuid: str) -> None:
    _call_users[call_id] = clone_user_uuid


def get_call_user(call_id: int) -> str | None:
    return _call_users.get(call_id)


def save_session(session: WebRTCSession) -> None:
    _sessions[session.call_id] = session


def get_session(call_id: int) -> WebRTCSession | None:
    return _sessions.get(call_id)


async def close_session(call_id: int) -> None:
    _call_users.pop(call_id, None)
    session = _sessions.pop(call_id, None)
    if session:
        for task in (session.receiver_task, session.pipeline_task):
            if task and not task.done():
                task.cancel()
        await session.peer_connection.close()
