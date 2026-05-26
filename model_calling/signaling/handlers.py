import asyncio
import json
from typing import Any

from model_calling.repository.clone_repository import (
    CloneNotFound,
    CloneRepositoryError,
    CloneRepositoryNotConfigured,
    find_clone_by_user_uuid,
)


async def handle_signaling_message(ws: Any, message: dict[str, Any]) -> None:
    message_type = message.get("type")

    if message_type == "CALL_INVITE":
        await handle_call_invite(ws, message)
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


async def send_json(ws: Any, message: dict[str, Any]) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))
