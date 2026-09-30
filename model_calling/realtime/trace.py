from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CallTrace:
    call_id: int
    user_id: str
    clone_id: int
    media_type: str
    started_at: float = field(default_factory=time.monotonic)
    turns_started: int = 0
    turns_completed: int = 0
    turns_failed: int = 0
    turns_skipped: int = 0
    last_error: str | None = None
    _active_turns: set[int] = field(default_factory=set, repr=False)
    _closed: bool = field(default=False, repr=False)

    def start_turn(self) -> int:
        self.turns_started += 1
        turn_id = self.turns_started
        self._active_turns.add(turn_id)
        return turn_id

    def finish_turn(
        self,
        turn_id: int,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        if turn_id not in self._active_turns:
            return
        self._active_turns.remove(turn_id)
        normalized = status.upper()
        if normalized == "COMPLETED":
            self.turns_completed += 1
        elif normalized == "SKIPPED":
            self.turns_skipped += 1
        else:
            self.turns_failed += 1
            self.last_error = error

    def close(self, *, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        cancelled = len(self._active_turns)
        self._active_turns.clear()
        duration_ms = round((time.monotonic() - self.started_at) * 1000)
        failed_close = "FAILED" in reason.upper() or "ERROR" in reason.upper()
        status = (
            "FAILED"
            if failed_close
            else "COMPLETED_WITH_ERRORS"
            if self.turns_failed or cancelled
            else "COMPLETED"
        )
        print(
            "[CALL_TRACE] call closed: "
            f"callId={self.call_id} user={self.user_id} clone_id={self.clone_id} "
            f"media_type={self.media_type} status={status} reason={reason} "
            f"duration_ms={duration_ms} turns_started={self.turns_started} "
            f"turns_completed={self.turns_completed} "
            f"turns_failed={self.turns_failed} "
            f"turns_skipped={self.turns_skipped} turns_cancelled={cancelled} "
            f"last_error={self.last_error or 'none'}",
            flush=True,
        )


def trace_fields(call_id: int | None, turn_id: int | None = None) -> str:
    call_value = str(call_id) if call_id is not None else "unknown"
    fields = f"callId={call_value}"
    if turn_id is not None:
        fields += f" turn={turn_id}"
    return fields
