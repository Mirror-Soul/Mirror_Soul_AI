import asyncio
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI

from model_calling.signaling.client import signaling_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    signaling_task = asyncio.create_task(signaling_loop())
    try:
        yield
    finally:
        signaling_task.cancel()
        with suppress(asyncio.CancelledError):
            await signaling_task


app = FastAPI(
    title="Mirror Soul Call Server",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "server": "call",
    }


if __name__ == "__main__":
    uvicorn.run(
        "call_server.main:app",
        host="0.0.0.0",
        port=8000,
    )
