"""Append-only live log of every call: full questions, answers and timings.

Unlike the dashboard (``realtime_call_monitor.py``) this never redraws or
cuts text. Each call and each turn gets a header, the user's words and the
clone's answer are printed in full, and everything is also saved as plain
text under ``tmp/monitor-logs`` so it can be shared after a test.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import threading

try:
    from tools.ai_pipeline_stream import StreamWorker, _ssh_args
    from tools.monitor_format import (
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_GREEN,
        ANSI_RED,
        ANSI_RESET,
        ANSI_WHITE,
        ANSI_YELLOW,
        display_width,
        enable_windows_ansi,
        tagged_event,
        terminal_width,
        timing_summary,
        wrap_text,
    )
except ModuleNotFoundError:
    from ai_pipeline_stream import StreamWorker, _ssh_args
    from monitor_format import (
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_GREEN,
        ANSI_RED,
        ANSI_RESET,
        ANSI_WHITE,
        ANSI_YELLOW,
        display_width,
        enable_windows_ansi,
        tagged_event,
        terminal_width,
        timing_summary,
        wrap_text,
    )


CALL_MARKER_PATTERN = (
    r"\[(SIGNALING|WEBRTC|REALTIME|DITTO_CALL|CALL_TRACE|VIDEO_OUT)\]"
    r"|Traceback|Error|Exception"
)
NOISE_PATTERN = r"sending video frame"
JOURNAL_PREFIX = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:?\d{2}|Z))\s+\S+\s+[^:]+:\s?(?P<msg>.*)$"
)
CALL_ID = re.compile(r"callId['\"]?\s*[:=]\s*['\"]?(\d+)")
TURN = re.compile(r"\bturn=(\d+)")
MEDIA = re.compile(r"(?:mediaType|media_type)=(\w+)")
USER = re.compile(r"\buser=([0-9a-f]{8})")
SPOKEN = re.compile(r"elapsed_ms=(\d+):\s?(?P<text>.*)$")


def remote_command(history: int, scan_lines: int) -> str:
    unit = "journalctl -u mirror-soul-call.service"
    history_command = ""
    if history:
        history_command = (
            f"{unit} -n {scan_lines} --no-pager -o short-iso 2>/dev/null "
            f"| grep -E '{CALL_MARKER_PATTERN}' | grep -v '{NOISE_PATTERN}' "
            f"| tail -n {history}; "
        )
    return (
        "{ "
        + history_command
        + f"{unit} -f -n 0 -o short-iso 2>/dev/null "
        f"| grep --line-buffered -E '{CALL_MARKER_PATTERN}' "
        f"| grep --line-buffered -v '{NOISE_PATTERN}'; "
        "}"
    )


def split_journal_line(raw: str) -> tuple[str, str]:
    """Return (local HH:MM:SS, message) from a ``journalctl -o short-iso`` line."""
    match = JOURNAL_PREFIX.match(raw)
    if not match:
        return datetime.now().strftime("%H:%M:%S"), raw
    stamp = match.group("ts").replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", stamp):
        stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    try:
        moment = datetime.fromisoformat(stamp).astimezone()
    except ValueError:
        return datetime.now().strftime("%H:%M:%S"), match.group("msg")
    return moment.strftime("%H:%M:%S"), match.group("msg")


def _paint(text: str, style: str, color: bool) -> str:
    return f"{style}{text}{ANSI_RESET}" if color and style else text


def _event_style(message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in ("failed", "error", "reject", "unavailable", "traceback", "exception")):
        return ANSI_RED + ANSI_BOLD
    if "warning" in lowered or "skipped" in lowered or "retry" in lowered or "count=0" in lowered:
        return ANSI_YELLOW
    if any(word in lowered for word in ("completed", "connected", "ready", "accept")):
        return ANSI_GREEN
    return ANSI_DIM


class CallLogFormatter:
    """Turns raw call-server lines into a readable, sectioned transcript."""

    def __init__(self, *, color: bool, width: int | None = None) -> None:
        self.color = color
        self.width = width
        self._call_id: str | None = None
        self._turn: tuple[str, str] | None = None

    def _width(self) -> int:
        return self.width or terminal_width()

    def _rule(self, title: str, char: str, style: str) -> list[str]:
        width = min(self._width(), 110)
        bar = char * max(4, width - display_width(title) - 4)
        return [_paint(f"{char * 2} {title} {bar}", style, self.color)]

    def _speech(self, time_text: str, who: str, text: str, note: str, style: str) -> list[str]:
        prefix = f"{time_text} "
        label = f"{who:<6}"
        indent = len(prefix) + len(label) + 3
        parts = wrap_text(text, self._width() - indent - 1)
        lines = [
            f"{_paint(prefix, ANSI_DIM, self.color)}"
            f"{_paint(f'[{label}]', ANSI_BOLD + style, self.color)} "
            f"{_paint(parts[0], ANSI_BOLD + style, self.color)}"
        ]
        lines.extend(" " * indent + _paint(part, ANSI_BOLD + style, self.color) for part in parts[1:])
        if note:
            lines.append(" " * indent + _paint(note, ANSI_DIM, self.color))
        return lines

    def format(self, raw: str) -> list[str]:
        time_text, message = split_journal_line(raw.rstrip())
        if not message.strip():
            return []
        output: list[str] = []
        call = CALL_ID.search(message)
        call_id = call.group(1) if call else None

        if "CALL_INVITE" in message and call_id:
            self._call_id = call_id
            self._turn = None
            output += [""] + self._rule(
                f"CALL {call_id} invited  {time_text}", "=", ANSI_BOLD + ANSI_BRIGHT_CYAN
            )
        elif "CALL_ACCEPT sent" in message and call_id:
            media = MEDIA.search(message)
            user = USER.search(message)
            if self._call_id != call_id:
                self._call_id = call_id
                output += [""] + self._rule(
                    f"CALL {call_id}  {time_text}", "=", ANSI_BOLD + ANSI_BRIGHT_CYAN
                )
            output.append(
                _paint(
                    f"   accepted: {media.group(1) if media else '-'} call, "
                    f"clone user {user.group(1) + '…' if user else '-'}",
                    ANSI_BRIGHT_CYAN,
                    self.color,
                )
            )
            return output

        turn = TURN.search(message)
        if call_id and turn:
            key = (call_id, turn.group(1))
            if key != self._turn:
                self._turn = key
                output += [""] + self._rule(
                    f"call {call_id} · turn {turn.group(1)}  {time_text}", "-", ANSI_BOLD
                )

        spoken = SPOKEN.search(message)
        if spoken and "[REALTIME] STT user=" in message:
            seconds = int(spoken.group(1)) / 1000
            return output + self._speech(
                time_text, "USER", spoken.group("text"), f"(speech-to-text {seconds:.1f}s)", ANSI_WHITE
            )
        if spoken and "[REALTIME] LLM user=" in message:
            seconds = int(spoken.group(1)) / 1000
            return output + self._speech(
                time_text, "CLONE", spoken.group("text"), f"(reply generated in {seconds:.1f}s)", ANSI_GREEN
            )
        if "[CALL_TRACE] turn" in message and timing_summary(message):
            status = "completed" if "turn completed" in message else "failed" if "turn failed" in message else "skipped"
            style = ANSI_GREEN if status == "completed" else ANSI_RED + ANSI_BOLD
            line = f"{time_text} " + _paint(f"[TURN  ] {status}: {timing_summary(message)}", ANSI_BOLD + style, self.color)
            error = re.search(r"error=(\S+)", message)
            if error:
                line += _paint(f"  error={error.group(1)}", ANSI_RED + ANSI_BOLD, self.color)
            return output + [line]

        prefix = f"{time_text} "
        output.append(
            tagged_event(
                message,
                _event_style(message),
                color=self.color,
                width=self._width(),
                prefix=_paint(prefix, ANSI_DIM, self.color),
                prefix_width=len(prefix),
            )
        )
        if "call closed" in message:
            output += self._rule(f"CALL {call_id or '-'} closed", "=", ANSI_DIM)
            self._turn = None
        return output


_ANSI = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Continuously append every call event with full text."
    )
    parser.add_argument("--history", type=int, default=200, help="past lines to show first")
    parser.add_argument("--scan-lines", type=int, default=5000)
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--call-host", default="43.202.181.134")
    parser.add_argument("--call-user", default="ec2-user")
    parser.add_argument("--call-key", type=Path, default=repo_root / "mirrorsoul-call-key.pem")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=repo_root / "tmp" / "monitor-logs",
        help="plain-text copy of everything shown (default: tmp/monitor-logs)",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not shutil.which("ssh"):
        print("ERROR: OpenSSH client (ssh) was not found.", file=sys.stderr)
        return 2
    enable_windows_ansi()
    color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    save_file = None
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = args.save_dir / f"call-events-{datetime.now():%Y%m%d}.log"
        save_file = save_path.open("a", encoding="utf-8")

    events: queue.Queue[tuple[str, str]] = queue.Queue()
    stop = threading.Event()
    worker = StreamWorker(
        source="CALL",
        args=_ssh_args(
            host=args.call_host,
            user=args.call_user,
            key=args.call_key,
            port=22,
            command=remote_command(args.history, args.scan_lines),
        ),
        output=events,
        stop=stop,
    )
    formatter = CallLogFormatter(color=color)

    def emit(lines: list[str]) -> None:
        for line in lines:
            print(line, flush=True)
            if save_file is not None:
                save_file.write(strip_ansi(line) + "\n")
        if save_file is not None:
            save_file.flush()

    emit(
        [
            _paint("MIRROR SOUL - LIVE CALL LOG  (every call, full text)", ANSI_BOLD + ANSI_BRIGHT_CYAN, color),
            "Nothing is cut or redrawn: scroll up to read earlier turns. Ctrl+C to stop.",
            _paint(f"Saved to: {save_file.name}", ANSI_DIM, color) if save_file else "Saving disabled.",
            "=" * min(terminal_width(), 110),
        ]
    )
    worker.start()
    try:
        while True:
            try:
                source, line = events.get(timeout=1)
            except queue.Empty:
                continue
            if source == "SYSTEM":
                emit([_paint(f"{datetime.now():%H:%M:%S} [SYSTEM] {line}", ANSI_RED, color)])
                continue
            emit(formatter.format(line))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.terminate()
        if save_file is not None:
            save_file.close()
    print("\nLive call log stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
