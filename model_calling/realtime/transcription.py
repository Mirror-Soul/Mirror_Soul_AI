"""Low-latency OpenAI Realtime transcription with upload-STT fallback support."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import suppress
from typing import Any

from openai import AsyncOpenAI


class RealtimeTranscriptionError(RuntimeError):
    pass


class RealtimeTranscriber:
    def __init__(self, manager: Any, connection: Any) -> None:
        self._manager = manager
        self._connection = connection
        self._send_lock = asyncio.Lock()
        self._audio_buffer = bytearray()
        self._pending_commits: deque[asyncio.Future[str]] = deque()
        self._items: dict[str, asyncio.Future[str]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def connect(
        cls,
        *,
        api_key: str,
        connection_model: str,
        model: str,
        language: str,
        timeout_seconds: float,
        client: AsyncOpenAI | None = None,
    ) -> "RealtimeTranscriber":
        realtime_client = client or AsyncOpenAI(api_key=api_key)
        manager = realtime_client.realtime.connect(model=connection_model)
        try:
            connection = await asyncio.wait_for(
                manager.__aenter__(),
                timeout=timeout_seconds,
            )
            transcriber = cls(manager, connection)
            await connection.session.update(
                session={
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "noise_reduction": {"type": "near_field"},
                            "transcription": {
                                "model": model,
                                "language": language,
                            },
                            "turn_detection": None,
                        }
                    },
                }
            )
            transcriber._reader_task = asyncio.create_task(
                transcriber._read_events()
            )
            return transcriber
        except Exception:
            with suppress(Exception):
                await manager.__aexit__(None, None, None)
            raise

    async def append_audio(self, pcm_24khz: bytes) -> None:
        if self._closed or not pcm_24khz:
            return
        self._audio_buffer.extend(pcm_24khz)
        # Send roughly every 100 ms to avoid a WebSocket write per 20 ms frame.
        if len(self._audio_buffer) >= 4_800:
            await self._flush_audio()

    async def clear(self) -> None:
        if self._closed:
            return
        self._audio_buffer.clear()
        async with self._send_lock:
            await self._connection.input_audio_buffer.clear()

    async def commit(self) -> asyncio.Future[str]:
        if self._closed:
            raise RealtimeTranscriptionError("Realtime transcription is closed.")
        await self._flush_audio()
        future = asyncio.get_running_loop().create_future()
        self._pending_commits.append(future)
        try:
            async with self._send_lock:
                await self._connection.input_audio_buffer.commit()
        except Exception:
            self._pending_commits.remove(future)
            raise
        return future

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
        self._fail_all(RealtimeTranscriptionError("Realtime transcription closed."))
        with suppress(Exception):
            await self._manager.__aexit__(None, None, None)

    async def _flush_audio(self) -> None:
        if not self._audio_buffer:
            return
        encoded = base64.b64encode(bytes(self._audio_buffer)).decode("ascii")
        self._audio_buffer.clear()
        async with self._send_lock:
            await self._connection.input_audio_buffer.append(audio=encoded)

    async def _read_events(self) -> None:
        try:
            async for event in self._connection:
                event_type = getattr(event, "type", "")
                if event_type == "input_audio_buffer.committed":
                    if self._pending_commits:
                        self._items[event.item_id] = self._pending_commits.popleft()
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    future = self._items.pop(event.item_id, None)
                    if future is not None and not future.done():
                        future.set_result(str(event.transcript or "").strip())
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    future = self._items.pop(event.item_id, None)
                    if future is not None and not future.done():
                        future.set_exception(
                            RealtimeTranscriptionError("Realtime transcription failed.")
                        )
                elif event_type == "error":
                    self._fail_all(
                        RealtimeTranscriptionError("Realtime transcription API error.")
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_all(
                RealtimeTranscriptionError(
                    f"Realtime transcription connection failed ({type(exc).__name__})."
                )
            )

    def _fail_all(self, error: Exception) -> None:
        futures = [*self._pending_commits, *self._items.values()]
        self._pending_commits.clear()
        self._items.clear()
        for future in futures:
            if not future.done():
                future.set_exception(error)
