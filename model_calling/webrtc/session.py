from dataclasses import dataclass

from aiortc import RTCPeerConnection


@dataclass
class WebRTCSession:
    call_id: int
    room_id: str
    ai_signal_id: str
    caller_signal_id: str
    peer_connection: RTCPeerConnection


_sessions: dict[int, WebRTCSession] = {}


def save_session(session: WebRTCSession) -> None:
    _sessions[session.call_id] = session


def get_session(call_id: int) -> WebRTCSession | None:
    return _sessions.get(call_id)


async def close_session(call_id: int) -> None:
    session = _sessions.pop(call_id, None)
    if session:
        await session.peer_connection.close()