from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from model_calling.webrtc.session import (
    WebRTCSession,
    get_session,
    save_session,
)


RTC_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    ]
)


def create_peer_connection() -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=RTC_CONFIG)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        print(f"[WEBRTC] connection state: {pc.connectionState}", flush=True)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"[WEBRTC] ice state: {pc.iceConnectionState}", flush=True)

    @pc.on("track")
    def on_track(track):
        print(f"[WEBRTC] track received: kind={track.kind}", flush=True)

    return pc


async def create_answer_from_offer(
    call_id: int,
    room_id: str,
    ai_signal_id: str,
    caller_signal_id: str,
    offer_sdp: dict,
) -> dict:
    session = get_session(call_id)

    if session is None:
        pc = create_peer_connection()
        session = WebRTCSession(
            call_id=call_id,
            room_id=room_id,
            ai_signal_id=ai_signal_id,
            caller_signal_id=caller_signal_id,
            peer_connection=pc,
        )
        save_session(session)

    pc = session.peer_connection

    offer = RTCSessionDescription(
        sdp=offer_sdp["sdp"],
        type=offer_sdp["type"],
    )

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp,
    }


async def create_offer_for_renegotiation(call_id: int) -> dict:
    session = get_session(call_id)
    if session is None:
        raise ValueError(f"WebRTC session not found: callId={call_id}")

    pc = session.peer_connection

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    return {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp,
    }


async def apply_answer(call_id: int, answer_sdp: dict) -> None:
    session = get_session(call_id)
    if session is None:
        raise ValueError(f"WebRTC session not found: callId={call_id}")

    answer = RTCSessionDescription(
        sdp=answer_sdp["sdp"],
        type=answer_sdp["type"],
    )

    await session.peer_connection.setRemoteDescription(answer)