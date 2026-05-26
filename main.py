import asyncio
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_calling.routers import chat
from model_calling.signaling.client import signaling_loop
from model_training.routers.training import router as training_router

app = FastAPI(title="Mirror Soul AI Server")

# 백엔드 시그널링 서버에 상시 접속하는 WebSocket 클라이언트 태스크
signaling_task = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_calling/assets 폴더가 없으면 자동 생성
ASSETS_DIR = os.path.join("model_calling", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# model_calling 라우터 연결
app.include_router(chat.router, prefix="/api/v1")

# model_training 라우터 연결
app.include_router(training_router)

@app.on_event("startup")
async def startup_event():
    global signaling_task
    # FastAPI 서버가 시작되면 백그라운드에서 시그널링 서버 접속/JOIN/재접속 루프를 실행
    signaling_task = asyncio.create_task(signaling_loop())


@app.on_event("shutdown")
async def shutdown_event():
    if signaling_task:
        # 서버 종료 시 백그라운드 WebSocket 루프도 함께 정리
        signaling_task.cancel()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
