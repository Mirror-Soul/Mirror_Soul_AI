import asyncio
import json
import time
from typing import Any

from model_calling.clients.backend_call_context import (
    CallContextError,
    fetch_call_context,
)

from model_calling.webrtc.peer import (
    add_remote_ice_candidate,
    apply_answer,
    create_answer_from_offer,
    create_offer_for_renegotiation,
)
from model_calling.webrtc.session import (
    close_session,
    get_call_context,
    register_call_context,
)


async def _preload_call_static_context(context: Any) -> None:
    """Prepare immutable persona and voice data before the first utterance."""
    from model_calling.realtime.pipeline import load_runtime_context

    started_at = time.monotonic()
    try:
        await load_runtime_context(
            context.clone.userUuid,
            context.clone.cloneId,
            call_id=context.callId,
        )
    except Exception as exc:
        # Keep accepting the call. The regular turn pipeline will retry and
        # report a stage-specific failure if this optional warm-up failed.
        print(
            "[REALTIME] call static context preload failed: "
            f"callId={context.callId} error={type(exc).__name__}",
            flush=True,
        )
        return
    elapsed_ms = max(0, round((time.monotonic() - started_at) * 1000))
    print(
        "[REALTIME] call static context preloaded: "
        f"callId={context.callId} user={context.clone.userUuid} "
        f"clone_id={context.clone.cloneId} elapsed_ms={elapsed_ms}",
        flush=True,
    )


async def _prewarm_call_dependencies(context: Any) -> None:
    from model_calling.realtime.prewarm import prewarm_realtime_dependencies

    try:
        await prewarm_realtime_dependencies(context.callId)
    except Exception as exc:
        print(
            "[REALTIME] dependency prewarm failed; continuing: "
            f"callId={context.callId} error={type(exc).__name__}",
            flush=True,
        )


async def handle_signaling_message(ws: Any, message: dict[str, Any]) -> None:
    message_type = message.get("type")

    if message_type == "CALL_INVITE":
        await handle_call_invite(ws, message)
        return

    if message_type == "OFFER":
        await handle_offer(ws, message)
        return

    if message_type == "ANSWER":
        await handle_answer(message)
        return

    if message_type == "ICE":
        await handle_ice(message)
        return

    if message_type == "CALL_END":
        await handle_call_end(message)
        return

    if message_type == "JOINED":
        return

    print(f"[SIGNALING] unsupported message type: {message_type}")


async def handle_call_invite(ws: Any, message: dict[str, Any]) -> None:
    raw_data = message.get("data")
    data = raw_data if isinstance(raw_data, dict) else {}
    call_id = data.get("callId")
    room_id = message.get("roomId")

    if isinstance(call_id, bool) or not isinstance(call_id, int) or call_id <= 0:
        await send_call_reject(
            ws,
            message,
            reason="INVALID_CALL_INVITE",
            detail="callId가 올바르지 않습니다.",
        )
        return

    context = get_call_context(call_id)
    is_new_context = context is None
    if context is None:
        try:
            context = await fetch_call_context(call_id)
        except CallContextError as exc:
            await send_call_reject(
                ws,
                message,
                reason=exc.reject_reason,
                detail=str(exc),
            )
            return

    if context.callId != call_id or context.roomId != room_id:
        await send_call_reject(
            ws,
            message,
            reason="CALL_CONTEXT_MISMATCH",
            detail="통화 컨텍스트의 callId 또는 roomId가 일치하지 않습니다.",
        )
        return

    register_call_context(context)
    if is_new_context:
        await asyncio.gather(
            _preload_call_static_context(context),
            _prewarm_call_dependencies(context),
        )

    accept_message = {
        "type": "CALL_ACCEPT",
        "roomId": room_id,
        "from": message.get("to"),
        "to": message.get("from"),
        "data": {
            "callId": call_id,
        },
    }

    try:
        await send_json(ws, accept_message)
    except Exception:
        await close_session(call_id, reason="CALL_ACCEPT_SEND_FAILED")
        raise
    print(
        "[SIGNALING] CALL_ACCEPT sent: "
        f"callId={call_id} mediaType={context.mediaType} "
        f"user={context.clone.userUuid} clone_id={context.clone.cloneId}",
        flush=True,
    )


async def send_call_reject(
    ws: Any,
    message: dict[str, Any],
    reason: str,
    detail: str,
) -> None:
    raw_data = message.get("data")
    data = raw_data if isinstance(raw_data, dict) else {}
    reject_message = {
        "type": "CALL_REJECT",
        "roomId": message.get("roomId"),
        "from": message.get("to"),
        "to": message.get("from"),
        "data": {
            "callId": data.get("callId"),
            "reason": reason,
            "detail": detail,
        },
    }

    await send_json(ws, reject_message)
    print(
        "[SIGNALING] CALL_REJECT sent: "
        f"callId={data.get('callId')} reason={reason}",
        flush=True,
    )


async def handle_offer(ws: Any, message: dict[str, Any]) -> None:
    data = message.get("data") or {}

    call_id = data.get("callId")
    offer_sdp = data.get("sdp")

    if not call_id or not offer_sdp:
        print("[SIGNALING] invalid OFFER message")
        return

    answer_sdp = await create_answer_from_offer(
        call_id=call_id,
        room_id=message.get("roomId"),
        ai_signal_id=message.get("to"),
        caller_signal_id=message.get("from"),
        offer_sdp=offer_sdp,
    )

    answer_message = {
        "type": "ANSWER",
        "roomId": message.get("roomId"),
        "from": message.get("to"),
        "to": message.get("from"),
        "data": {
            "callId": call_id,
            "sdp": answer_sdp,
        },
    }

    await send_json(ws, answer_message)
    print(f"[SIGNALING] ANSWER sent: callId={call_id}", flush=True)


async def handle_answer(message: dict[str, Any]) -> None:
    data = message.get("data") or {}

    call_id = data.get("callId")
    answer_sdp = data.get("sdp")

    if not call_id or not answer_sdp:
        print("[SIGNALING] invalid ANSWER message")
        return

    await apply_answer(call_id, answer_sdp)
    print(f"[SIGNALING] ANSWER applied: callId={call_id}", flush=True)


async def send_offer(ws: Any, call_id: int) -> None:
    offer_sdp = await create_offer_for_renegotiation(call_id)

    from model_calling.webrtc.session import get_session

    session = get_session(call_id)
    if session is None:
        raise ValueError(f"WebRTC session not found: callId={call_id}")

    offer_message = {
        "type": "OFFER",
        "roomId": session.room_id,
        "from": session.ai_signal_id,
        "to": session.caller_signal_id,
        "data": {
            "callId": call_id,
            "sdp": offer_sdp,
        },
    }

    await send_json(ws, offer_message)
    print(f"[SIGNALING] OFFER sent: callId={call_id}", flush=True)


async def handle_ice(message: dict[str, Any]) -> None:
    data = message.get("data") or {}
    call_id = data.get("callId")

    if call_id is None:
        print("[SIGNALING] invalid ICE: callId is required.", flush=True)
        return

    if "candidate" not in data:
        print("[SIGNALING] invalid ICE: candidate field is required.", flush=True)
        return

    try:
        await add_remote_ice_candidate(
            call_id=call_id,
            candidate_data=data["candidate"],
        )
    except (AssertionError, KeyError, ValueError) as exc:
        print(f"[SIGNALING] ICE handling failed: {exc}", flush=True)


async def handle_call_end(message: dict[str, Any]) -> None:
    data = message.get("data") or {}
    call_id = data.get("callId")
    if call_id is None:
        print("[SIGNALING] invalid CALL_END: callId is required.", flush=True)
        return

    await close_session(call_id, reason="CALL_END")
    print(f"[SIGNALING] CALL_END handled: callId={call_id}", flush=True)


async def send_json(ws: Any, message: dict[str, Any]) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))
