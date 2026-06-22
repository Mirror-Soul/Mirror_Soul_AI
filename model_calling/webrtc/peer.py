import os

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp
from dotenv import load_dotenv

from model_calling.webrtc.session import (
    WebRTCSession,
    get_session,
    save_session,
)

load_dotenv()


def create_rtc_configuration() -> RTCConfiguration:
    ice_servers = [
        RTCIceServer(
            urls=[
                os.getenv(
                    "WEBRTC_STUN_URL",
                    "stun:stun.l.google.com:19302",
                )
            ]
        )
    ]

    turn_url = os.getenv("WEBRTC_TURN_URL")
    if turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=[turn_url],
                username=os.getenv("WEBRTC_TURN_USERNAME"),
                credential=os.getenv("WEBRTC_TURN_CREDENTIAL"),
            )
        )

    return RTCConfiguration(iceServers=ice_servers)


def create_peer_connection() -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=create_rtc_configuration())

    @pc.on("icegatheringstatechange")
    async def on_ice_gathering_state_change():
        print(f"[WEBRTC] ICE gathering: {pc.iceGatheringState}", flush=True)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        print(f"[WEBRTC] connection: {pc.connectionState}", flush=True)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"[WEBRTC] ICE connection: {pc.iceConnectionState}", flush=True)

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


async def add_remote_ice_candidate(
    call_id: int,
    candidate_data: dict | None,
) -> None:
    session = get_session(call_id)
    if session is None:
        raise ValueError(f"WebRTC session not found: callId={call_id}")

    pc = session.peer_connection

    # null candidate는 상대방의 ICE candidate 수집이 끝났다는 의미다.
    if candidate_data is None:
        await pc.addIceCandidate(None)
        print(f"[WEBRTC] remote ICE completed: callId={call_id}", flush=True)
        return

    candidate_text = candidate_data.get("candidate")
    if not candidate_text:
        raise ValueError("ICE candidate is required.")

    if candidate_text.startswith("candidate:"):
        candidate_text = candidate_text[len("candidate:"):]

    candidate = candidate_from_sdp(candidate_text)
    candidate.sdpMid = candidate_data.get("sdpMid")
    candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")

    if candidate.sdpMid is None and candidate.sdpMLineIndex is None:
        raise ValueError("sdpMid or sdpMLineIndex is required.")

    await pc.addIceCandidate(candidate)
    print(f"[WEBRTC] remote ICE added: callId={call_id}", flush=True)
