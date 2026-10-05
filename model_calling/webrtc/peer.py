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
    get_call_context,
    get_session,
    save_session,
)
from model_calling.realtime.audio import QueuedAudioTrack
from model_calling.realtime.ditto import (
    DittoRealtimeError,
    create_ditto_video_session,
)
from model_calling.realtime.video import QueuedVideoTrack

load_dotenv()


def create_rtc_configuration(call_id: int | None = None) -> RTCConfiguration:
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
        f"callId={call_id if call_id is not None else 'unknown'} "
        f"ice_servers={len(ice_servers)} "
        f"turn={'enabled' if turn_url else 'disabled'}",
        flush=True,
    )
    return RTCConfiguration(iceServers=ice_servers)


def create_peer_connection(call_id: int) -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=create_rtc_configuration(call_id))
    print(f"[WEBRTC] peer connection created: callId={call_id}", flush=True)

    @pc.on("icegatheringstatechange")
    async def on_ice_gathering_state_change():
        print(
            f"[WEBRTC] ICE gathering: callId={call_id} "
            f"state={pc.iceGatheringState}",
            flush=True,
        )

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        print(
            f"[WEBRTC] connection: callId={call_id} state={pc.connectionState}",
            flush=True,
        )
        if pc.connectionState in {"failed", "closed"}:
            await close_session(
                call_id,
                reason=f"PEER_{pc.connectionState.upper()}",
            )

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(
            f"[WEBRTC] ICE connection: callId={call_id} "
            f"state={pc.iceConnectionState}",
            flush=True,
        )

    @pc.on("track")
    def on_track(track):
        print(
            f"[WEBRTC] track received: callId={call_id} kind={track.kind}",
            flush=True,
        )
        if track.kind != "audio":
            print(
                f"[WEBRTC] non-audio track ignored: callId={call_id} "
                f"kind={track.kind}",
                flush=True,
            )
            return

        session = get_session(call_id)
        if session is None:
            print(f"[WEBRTC] track ignored because session is missing: callId={call_id}", flush=True)
            return
        if session.receiver_task is not None or session.pipeline_start_task is not None:
            print(f"[WEBRTC] duplicate audio track ignored: callId={call_id}", flush=True)
            return

        async def start_pipeline() -> None:
            from model_calling.realtime.pipeline import start_realtime_audio

            print(
                "[WEBRTC] starting realtime pipeline for track: "
                f"callId={call_id} user={session.clone_user_uuid} "
                f"clone_id={session.clone_id}",
                flush=True,
            )
            try:
                receiver_task, pipeline_task = await start_realtime_audio(
                    call_id=call_id,
                    user_id=session.clone_user_uuid,
                    clone_id=session.clone_id,
                    incoming_track=track,
                    output_track=session.output_track,
                    utterance_queue=session.utterance_queue,
                    video_renderer=session.video_renderer,
                    video_required=session.media_type == "VIDEO",
                    conversation_history=session.conversation_history,
                    call_trace=session.trace,
                )
                if get_session(call_id) is not session:
                    receiver_task.cancel()
                    pipeline_task.cancel()
                    await asyncio.gather(
                        receiver_task,
                        pipeline_task,
                        return_exceptions=True,
                    )
                    return
                session.receiver_task = receiver_task
                session.pipeline_task = pipeline_task
                print(
                    "[WEBRTC] realtime pipeline attached: "
                    f"callId={call_id} receiver_task={id(receiver_task)} "
                    f"pipeline_task={id(pipeline_task)}",
                    flush=True,
                )
            finally:
                session.pipeline_start_task = None

        session.pipeline_start_task = asyncio.create_task(start_pipeline())

    return pc


async def create_answer_from_offer(
    call_id: int,
    room_id: str,
    ai_signal_id: str,
    caller_signal_id: str,
    offer_sdp: dict,
) -> dict:
    context = get_call_context(call_id)
    if context is None:
        raise ValueError(f"Call context not registered: callId={call_id}")
    if context.roomId != room_id:
        raise ValueError(f"Call room does not match context: callId={call_id}")

    session = get_session(call_id)

    if session is None:
        clone_user_uuid = context.clone.userUuid
        clone_id = context.clone.cloneId
        media_type = context.mediaType

        pc = create_peer_connection(call_id)
        output_track = QueuedAudioTrack()
        pc.addTrack(output_track)
        print(f"[WEBRTC] output audio track added: callId={call_id}", flush=True)
        output_video_track = None
        video_renderer = None
        if media_type == "VIDEO":
            output_video_track = QueuedVideoTrack(
                width=int(os.getenv("REALTIME_VIDEO_WIDTH", "540")),
                height=int(os.getenv("REALTIME_VIDEO_HEIGHT", "960")),
                fps=int(os.getenv("REALTIME_VIDEO_FPS", "25")),
                # The zoom/pan on the still portrait drifts away from the
                # frame Ditto replies start on, which reads as a jump.
                idle_motion_enabled=os.getenv(
                    "REALTIME_IDLE_MOTION_ENABLED",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"},
                idle_motion_scale=float(
                    os.getenv("REALTIME_IDLE_MOTION_SCALE", "0.012")
                ),
                idle_motion_period_seconds=float(
                    os.getenv("REALTIME_IDLE_MOTION_PERIOD_SECONDS", "6.0")
                ),
                transition_frames=int(
                    os.getenv("REALTIME_VIDEO_TRANSITION_FRAMES", "6")
                ),
            )
            pc.addTrack(output_video_track)
            try:
                video_renderer = create_ditto_video_session(
                    call_id=call_id,
                    user_id=clone_user_uuid,
                    clone_id=clone_id,
                    track=output_video_track,
                )
            except DittoRealtimeError as exc:
                print(
                    "[WEBRTC] Ditto video renderer unavailable: "
                    f"callId={call_id} error={exc}",
                    flush=True,
                )
            print(
                "[WEBRTC] output video track added: "
                f"callId={call_id} renderer={'enabled' if video_renderer else 'disabled'}",
                flush=True,
            )
        session = WebRTCSession(
            call_id=call_id,
            room_id=room_id,
            ai_signal_id=ai_signal_id,
            caller_signal_id=caller_signal_id,
            peer_connection=pc,
            clone_user_uuid=clone_user_uuid,
            clone_id=clone_id,
            output_track=output_track,
            utterance_queue=asyncio.Queue(maxsize=2),
            media_type=media_type,
            output_video_track=output_video_track,
            video_renderer=video_renderer,
        )
        save_session(session)
        if video_renderer is not None:
            async def prepare_video() -> None:
                try:
                    await video_renderer.prepare()
                    print(
                        f"[WEBRTC] idle portrait ready: callId={call_id}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        "[WEBRTC] idle portrait preparation failed: "
                        f"callId={call_id} error={exc!r}",
                        flush=True,
                    )
                    return
                prepare_idle_loop = getattr(
                    video_renderer,
                    "prepare_idle_loop",
                    None,
                )
                if prepare_idle_loop is None:
                    return
                try:
                    await prepare_idle_loop()
                except Exception as exc:
                    print(
                        "[WEBRTC] idle loop preparation failed: "
                        f"callId={call_id} error={exc!r}",
                        flush=True,
                    )

            session.video_prepare_task = asyncio.create_task(prepare_video())
        print(
            "[WEBRTC] session created: "
            f"callId={call_id} roomId={room_id} "
            f"user={clone_user_uuid} clone_id={clone_id}",
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
