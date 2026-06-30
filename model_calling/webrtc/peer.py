import asyncio
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
    close_session,
    get_call_user,
    get_session,
    save_session,
)
from model_calling.realtime.audio import QueuedAudioTrack
from model_calling.realtime import start_realtime_audio

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

    print(
        "[WEBRTC] RTC configuration created: "
        f"ice_servers={len(ice_servers)} turn={'enabled' if turn_url else 'disabled'}",
        flush=True,
    )
    return RTCConfiguration(iceServers=ice_servers)


def create_peer_connection(call_id: int) -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=create_rtc_configuration())
    print(f"[WEBRTC] peer connection created: callId={call_id}", flush=True)

    @pc.on("icegatheringstatechange")
    async def on_ice_gathering_state_change():
        print(f"[WEBRTC] ICE gathering: {pc.iceGatheringState}", flush=True)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        print(f"[WEBRTC] connection: {pc.connectionState}", flush=True)
        if pc.connectionState in {"failed", "closed"}:
            await close_session(call_id)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"[WEBRTC] ICE connection: {pc.iceConnectionState}", flush=True)

    @pc.on("track")
    def on_track(track):
        print(f"[WEBRTC] track received: kind={track.kind}", flush=True)
        if track.kind != "audio":
            print(f"[WEBRTC] non-audio track ignored: kind={track.kind}", flush=True)
            return

        session = get_session(call_id)
        if session is None:
            print(f"[WEBRTC] track ignored because session is missing: callId={call_id}", flush=True)
            return
        if session.receiver_task is not None:
            print(f"[WEBRTC] duplicate audio track ignored: callId={call_id}", flush=True)
            return

        async def start_pipeline() -> None:
            print(
                "[WEBRTC] starting realtime pipeline for track: "
                f"callId={call_id} user={session.clone_user_uuid}",
                flush=True,
            )
            receiver_task, pipeline_task = await start_realtime_audio(
                user_id=session.clone_user_uuid,
                incoming_track=track,
                output_track=session.output_track,
                utterance_queue=session.utterance_queue,
            )
            session.receiver_task = receiver_task
            session.pipeline_task = pipeline_task
            print(
                "[WEBRTC] realtime pipeline attached: "
                f"callId={call_id} receiver_task={id(receiver_task)} "
                f"pipeline_task={id(pipeline_task)}",
                flush=True,
            )

        asyncio.create_task(start_pipeline())

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
        clone_user_uuid = get_call_user(call_id)
        if not clone_user_uuid:
            raise ValueError(f"Call user not registered: callId={call_id}")

        pc = create_peer_connection(call_id)
        output_track = QueuedAudioTrack()
        pc.addTrack(output_track)
        print(f"[WEBRTC] output audio track added: callId={call_id}", flush=True)
        session = WebRTCSession(
            call_id=call_id,
            room_id=room_id,
            ai_signal_id=ai_signal_id,
            caller_signal_id=caller_signal_id,
            peer_connection=pc,
            clone_user_uuid=clone_user_uuid,
            output_track=output_track,
            utterance_queue=asyncio.Queue(maxsize=2),
        )
        save_session(session)
        print(
            "[WEBRTC] session created: "
            f"callId={call_id} roomId={room_id} user={clone_user_uuid}",
            flush=True,
        )

    pc = session.peer_connection

    offer = RTCSessionDescription(
        sdp=offer_sdp["sdp"],
        type=offer_sdp["type"],
    )

    print(
        "[WEBRTC] applying remote offer: "
        f"callId={call_id} type={offer.type} sdp_length={len(offer.sdp)}",
        flush=True,
    )
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print(
        "[WEBRTC] local answer created: "
        f"callId={call_id} type={pc.localDescription.type} "
        f"sdp_length={len(pc.localDescription.sdp)}",
        flush=True,
    )

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
    print(
        "[WEBRTC] local offer created: "
        f"callId={call_id} type={pc.localDescription.type} "
        f"sdp_length={len(pc.localDescription.sdp)}",
        flush=True,
    )

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

    print(
        "[WEBRTC] applying remote answer: "
        f"callId={call_id} type={answer.type} sdp_length={len(answer.sdp)}",
        flush=True,
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
