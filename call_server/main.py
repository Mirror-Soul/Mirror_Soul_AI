import asyncio
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI

from model_calling.signaling.client import signaling_loop
from model_calling.clients.rag_search import search_mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    if search_mode() == "remote":
        print("[RAG_SEARCH] mode=remote (AI API server store)", flush=True)
    else:
        # The call server has no member memories of its own; without
        # RAG_SEARCH_BASE_URL every realtime RAG lookup returns nothing.
        print(
            "[RAG_SEARCH] WARNING mode=local: RAG_SEARCH_BASE_URL is not set, "
            "realtime calls will not see member memories.",
            flush=True,
        )
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
        "ragSearch": search_mode(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "call_server.main:app",
        host="0.0.0.0",
        port=8000,
    )
