"""Per-call recorder that sends finished utterances to the backend history.

Saving never blocks the conversation: entries are queued and a background
task posts them one at a time, in the order they were spoken. When the call
closes, ``close()`` waits a bounded time for the remaining entries.
"""

from __future__ import annotations

import asyncio
import io
import wave
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from model_calling.clients.backend_talk_log import (
    TalkLogEntry,
    TalkLogSaveError,
    TalkLogSaveResult,
    normalize_message,
    save_talk_log,
    talk_log_config_problem,
)
from model_calling.realtime.trace import trace_fields
from shared.config import settings


TalkLogSaver = Callable[[int, TalkLogEntry], Awaitable[TalkLogSaveResult]]

_WAV_HEADER_BYTES = 44
_DEFAULT_WAV_BYTES_PER_SECOND = 16_000 * 2


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Duration of a PCM WAV payload; falls back to 16 kHz mono PCM16."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            rate = reader.getframerate()
            if rate > 0:
                return reader.getnframes() / rate
    except (wave.Error, EOFError):
        pass
    payload = max(0, len(wav_bytes) - _WAV_HEADER_BYTES)
    return payload / _DEFAULT_WAV_BYTES_PER_SECOND


def spoken_window(
    *,
    ended_at: datetime,
    duration_seconds: float,
) -> tuple[datetime, datetime]:
    duration = max(0.0, float(duration_seconds))
    return ended_at - timedelta(seconds=duration), ended_at


class CallTalkLogRecorder:
    def __init__(
        self,
        call_id: int,
        *,
        saver: TalkLogSaver | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.call_id = call_id
        self._saver = saver or save_talk_log
        if enabled is None:
            problem = talk_log_config_problem()
            enabled = problem is None
            self.disabled_reason = problem
        else:
            self.disabled_reason = None if enabled else "disabled"
        self.enabled = enabled
        self._pending: list[TalkLogEntry] = []
        self._worker: asyncio.Task | None = None
        self._closed = False
        self._announced_disabled = False
        self.saved = 0
        self.duplicated = 0
        self.failed = 0
        self.dropped = 0

    @property
    def pending(self) -> int:
        return len(self._pending)

    def record_turn(
        self,
        *,
        turn_id: int | None,
        user_text: str,
        user_started_at: datetime,
        user_ended_at: datetime,
        clone_text: str,
        clone_started_at: datetime,
        clone_ended_at: datetime | None,
    ) -> None:
        self.record(
            TalkLogEntry(
                speaker="USER",
                message=user_text,
                started_at=user_started_at,
                ended_at=user_ended_at,
                turn_id=turn_id,
            )
        )
        self.record(
            TalkLogEntry(
                speaker="CLONE",
                message=clone_text,
                started_at=clone_started_at,
                ended_at=clone_ended_at,
                turn_id=turn_id,
            )
        )

    def record(self, entry: TalkLogEntry) -> None:
        if not self.enabled:
            if not self._announced_disabled:
                self._announced_disabled = True
                print(
                    "[TALK_LOG] saving disabled: "
                    f"callId={self.call_id} reason={self.disabled_reason}",
                    flush=True,
                )
            return
        if self._closed:
            self.dropped += 1
            print(
                "[TALK_LOG] entry dropped after close: "
                f"{trace_fields(self.call_id, entry.turn_id)} speaker={entry.speaker}",
                flush=True,
            )
            return
        if not normalize_message(entry.message):
            return
        self._pending.append(entry)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._pending:
            entry = self._pending[0]
            fields = trace_fields(self.call_id, entry.turn_id)
            try:
                result = await self._saver(self.call_id, entry)
            except asyncio.CancelledError:
                raise
            except TalkLogSaveError as exc:
                self.failed += 1
                print(
                    "[TALK_LOG] save failed: "
                    f"{fields} speaker={entry.speaker} "
                    f"error_code={exc.code} error={exc}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - never break the call
                self.failed += 1
                print(
                    "[TALK_LOG] save failed: "
                    f"{fields} speaker={entry.speaker} "
                    f"error_code=TALK_LOG_UNEXPECTED error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                if result.duplicated:
                    self.duplicated += 1
                else:
                    self.saved += 1
                print(
                    "[TALK_LOG] saved: "
                    f"{fields} speaker={entry.speaker} "
                    f"talkLogId={result.talk_log_id} "
                    f"duplicated={'yes' if result.duplicated else 'no'} "
                    f"chars={len(normalize_message(entry.message))}",
                    flush=True,
                )
            finally:
                if self._pending and self._pending[0] is entry:
                    self._pending.pop(0)

    async def close(self, timeout_seconds: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        timeout = (
            settings.BACKEND_TALK_LOG_FLUSH_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        worker = self._worker
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(asyncio.shield(worker), max(0.0, timeout))
            except asyncio.TimeoutError:
                self.dropped += len(self._pending)
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                self._pending.clear()
        print(
            "[TALK_LOG] call summary: "
            f"callId={self.call_id} saved={self.saved} "
            f"duplicated={self.duplicated} failed={self.failed} "
            f"dropped={self.dropped}",
            flush=True,
        )
