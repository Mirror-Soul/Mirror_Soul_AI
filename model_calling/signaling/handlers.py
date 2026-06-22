import asyncio
import json
from typing import Any

from model_calling.repository.clone_repository import (
    CloneNotFound,
    CloneRepositoryError,
    CloneRepositoryNotConfigured,
    find_clone_by_user_uuid,
)

from model_calling.webrtc.peer import (
    add_remote_ice_candidate,
    apply_answer,
    create_answer_from_offer,
    create_offer_for_renegotiation,
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

    print(f"[SIGNALING] unsupported message type: {message_type}")


async def handle_call_invite(ws: Any, message: dict[str, Any]) -> None:
    data = message.get("data") or {}
    call_id = data.get("callId")
    clone_user_uuid = data.get("cloneUserUuid")
    media_type = data.get("mediaType")

    if not call_id or not clone_user_uuid or not media_type:
        await send_call_reject(
            ws,
            message,
            reason="INVALID_CALL_INVITE",
            detail="통화 초대 메시지 형식이 올바르지 않습니다.",
        )
        return

    try:
        # RDS 접근은 동기 DB 드라이버를 사용하므로 이벤트 루프를 막지 않도록 별도 스레드에서 실행한다.
        clone_info = await asyncio.to_thread(
            find_clone_by_user_uuid,
            clone_user_uuid,
        )
    except CloneNotFound:
        await send_call_reject(
            ws,
            message,
            reason="CLONE_NOT_FOUND",
            detail="클론 정보를 찾을 수 없습니다.",
        )
        return
    except CloneRepositoryNotConfigured:
        await send_call_reject(
            ws,
            message,
            reason="RDS_NOT_CONFIGURED",
            detail="RDS 연결 설정이 완료되지 않았습니다.",
        )
        return
    except CloneRepositoryError:
        await send_call_reject(
            ws,
            message,
            reason="RDS_LOOKUP_FAILED",
            detail="클론 정보 조회 중 오류가 발생했습니다.",
        )
        return

    accept_message = {
        "type": "CALL_ACCEPT",
        "roomId": message.get("roomId"),
        "from": message.get("to"),
        "to": message.get("from"),
        "data": {
            "callId": call_id,
            "cloneUserUuid": clone_info.clone_user_uuid,
            "mediaType": media_type,
        },
    }

    await send_json(ws, accept_message)
    print(f"[SIGNALING] CALL_ACCEPT sent: callId={call_id}")


async def send_call_reject(
    ws: Any,
    message: dict[str, Any],
    reason: str,
    detail: str,
) -> None:
    data = message.get("data") or {}
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
    print(f"[SIGNALING] CALL_REJECT sent: reason={reason}")

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


async def send_json(ws: Any, message: dict[str, Any]) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))

