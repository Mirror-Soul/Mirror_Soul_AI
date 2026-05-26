import asyncio
import json
import websockets

from model_calling.signaling.handlers import handle_signaling_message

SIGNALING_URL = "wss://api.mirrorsoul64.com/ws/signaling"
AI_SIGNAL_ID = "ai-server"


async def signaling_loop():
    # AI 서버는 백엔드 시그널링 서버에 상시 연결되어 있어야 하므로 끊기면 계속 재접속
    while True:
        try:
            print("[SIGNALING] connecting...")

            async with websockets.connect(SIGNALING_URL) as ws:
                print("[SIGNALING] connected")

                # 백엔드는 ai-server 세션으로 JOIN한 연결에 프론트의 AI 대상 메시지를 전달
                join_message = {
                    "type": "JOIN",
                    "roomId": None,
                    "from": AI_SIGNAL_ID,
                    "to": "server",
                    "data": None,
                }

                await ws.send(json.dumps(join_message))
                print("[SIGNALING] JOIN sent")

                async for raw_message in ws:
                    message = json.loads(raw_message)
                    print("[SIGNALING] received:", message)

                    # CALL_INVITE 수신 시 RDS에서 클론 정보를 조회한 뒤 ACCEPT/REJECT를 응답한다.
                    await handle_signaling_message(ws, message)

        except Exception as e:
            print("[SIGNALING] disconnected:", e)
            print("[SIGNALING] reconnecting in 3 seconds...")
            await asyncio.sleep(3)
