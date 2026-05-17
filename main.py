import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_calling.routers import chat
from model_training.routers.training import router as training_router

app = FastAPI(title="Mirror Soul AI Server")

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)