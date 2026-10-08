"""Append-only live log of every call, in Korean: full questions, answers and timings.

Unlike the dashboard (``realtime_call_monitor.py``) this never redraws or
cuts text. Each call and each turn gets a header, the user's words and the
clone's answer are printed in full, and everything is also saved as plain
text under ``tmp/monitor-logs`` so it can be shared after a test.
"""

from __future__ import annotations

import argparse
from collections import deque
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
    from tools.call_event_text import describe, turn_summary_ko
    from tools.monitor_format import (
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_GREEN,
        ANSI_RED,
        ANSI_RESET,
        ANSI_WHITE,
        ANSI_YELLOW,
        compact_event,
        display_width,
        event_tag,
        enable_windows_ansi,
        tagged_event,
        terminal_width,
        timing_summary,
        wrap_text,
    )
except ModuleNotFoundError:
    from ai_pipeline_stream import StreamWorker, _ssh_args
    from call_event_text import describe, turn_summary_ko
    from monitor_format import (
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_GREEN,
        ANSI_RED,
        ANSI_RESET,
        ANSI_WHITE,
        ANSI_YELLOW,
        compact_event,
        display_width,
        event_tag,
        enable_windows_ansi,
        tagged_event,
        terminal_width,
        timing_summary,
        wrap_text,
    )


CALL_MARKER_PATTERN = (
    r"\[(SIGNALING|WEBRTC|REALTIME|DITTO_CALL|CALL_TRACE|VIDEO_OUT|TALK_LOG)\]"
    r"|Traceback|Error|Exception"
)
NOISE_PATTERN = r"sending video frame"
LIVE_MARKER = "__MIRROR_SOUL_LIVE__"
JOURNAL_PREFIX = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:?\d{2}|Z))\s+\S+\s+[^:]+:\s?(?P<msg>.*)$"
)
CALL_ID = re.compile(r"callId['\"]?\s*[:=]\s*['\"]?(\d+)")
TURN = re.compile(r"\bturn=(\d+)")
SPOKEN = re.compile(r"elapsed_ms=(\d+):\s?(?P<text>.*)$")


def remote_command(history_minutes: int, scan_lines: int) -> str:
    """Recent history (if any), a live marker, then the live tail."""
    unit = "journalctl -u mirror-soul-call.service"
    history_command = ""
    if history_minutes > 0:
        history_command = (
            f"{unit} --since '-{history_minutes} minutes' --no-pager -o short-iso 2>/dev/null "
            f"| grep -E '{CALL_MARKER_PATTERN}' | grep -v '{NOISE_PATTERN}' "
            f"| tail -n {scan_lines}; "
        )
    return (
        "{ "
        + history_command
        + f"echo {LIVE_MARKER}; "
        + f"{unit} -f -n 0 -o short-iso 2>/dev/null "
        f"| grep --line-buffered -E '{CALL_MARKER_PATTERN}' "
        f"| grep --line-buffered -v '{NOISE_PATTERN}'; "
        "}"
    )


def parse_journal_line(raw: str) -> tuple[datetime | None, str]:
    """Return (local time or None, message) from a ``journalctl -o short-iso`` line."""
    match = JOURNAL_PREFIX.match(raw)
    if not match:
        return None, raw
    stamp = match.group("ts").replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", stamp):
        stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    try:
        return datetime.fromisoformat(stamp).astimezone(), match.group("msg")
    except ValueError:
        return None, match.group("msg")


def split_journal_line(raw: str) -> tuple[str, str]:
    """Return (local HH:MM:SS, message) from a ``journalctl -o short-iso`` line."""
    moment, message = parse_journal_line(raw)
    return (moment or datetime.now()).strftime("%H:%M:%S"), message


def _paint(text: str, style: str, color: bool) -> str:
    return f"{style}{text}{ANSI_RESET}" if color and style else text


def _label(tag: str) -> str:
    return f"[{tag}{' ' * max(0, 6 - display_width(tag))}]"


def _fallback_style(message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in ("failed", "error", "reject", "unavailable", "traceback", "exception")):
        return ANSI_RED + ANSI_BOLD
    if "warning" in lowered or "skipped" in lowered or "retry" in lowered:
        return ANSI_YELLOW
    return ANSI_DIM


class CallLogFormatter:
    """Turns raw call-server lines into a readable Korean transcript."""

    def __init__(self, *, color: bool, width: int | None = None) -> None:
        self.color = color
        self.width = width
        self._call_id: str | None = None
        self._turn: tuple[str, str] | None = None
        self._date: str | None = None
        self._recent: deque[tuple[str, str]] = deque(maxlen=3)
        self._suppressed = 0
        self._history_lines = 0
        self._live = False

    def _width(self) -> int:
        return self.width or terminal_width()

    def _rule(self, title: str, char: str, style: str) -> list[str]:
        width = min(self._width(), 110)
        bar = char * max(4, width - display_width(title) - 4)
        return [_paint(f"{char * 2} {title} {bar}", style, self.color)]

    def _flush_repeats(self) -> list[str]:
        if not self._suppressed:
            return []
        count, self._suppressed = self._suppressed, 0
        return [" " * 18 + _paint(f"(같은 내용 {count}줄 더 반복)", ANSI_DIM, self.color)]

    def _reset_repeats(self) -> list[str]:
        lines = self._flush_repeats()
        self._recent.clear()
        return lines

    def _line(self, clock: str, tag: str, text: str, style: str, *, bold: bool = False) -> list[str]:
        prefix = f"{clock} "
        indent = len(prefix) + 9
        parts = wrap_text(text, self._width() - indent - 1)
        weight = ANSI_BOLD if bold else ""
        lines = [
            _paint(prefix, ANSI_DIM, self.color)
            + _paint(_label(tag), ANSI_BOLD + style, self.color)
            + " "
            + _paint(parts[0], weight + style, self.color)
        ]
        lines.extend(" " * indent + _paint(part, weight + style, self.color) for part in parts[1:])
        return lines

    def live_started(self) -> list[str]:
        self._live = True
        lines = self._reset_repeats()
        if self._history_lines == 0:
            lines.append(_paint("(최근 기록 없음)", ANSI_DIM, self.color))
        return lines + [""] + self._rule("여기부터 실시간 로그", "#", ANSI_BOLD + ANSI_BRIGHT_CYAN)

    def format(self, raw: str) -> list[str]:
        if raw.strip() == LIVE_MARKER:
            return self.live_started()
        moment, message = parse_journal_line(raw.rstrip())
        message = message.strip()
        if not message:
            return []
        if not self._live:
            self._history_lines += 1
        moment = moment or datetime.now()
        clock = moment.strftime("%H:%M:%S")
        output: list[str] = []

        day = moment.strftime("%Y-%m-%d")
        if day != self._date:
            self._date = day
            today = datetime.now().strftime("%Y-%m-%d")
            label = "오늘" if day == today else "이전 기록"
            output += self._reset_repeats() + self._rule(f"{day} ({label})", "~", ANSI_DIM)

        call = CALL_ID.search(message)
        call_id = call.group(1) if call else None
        if "'type': 'CALL_INVITE'" in message and call_id:
            self._call_id, self._turn = call_id, None
            output += self._reset_repeats() + [""] + self._rule(
                f"통화 {call_id} 요청  {clock}", "=", ANSI_BOLD + ANSI_BRIGHT_CYAN
            )
        elif "CALL_ACCEPT sent" in message and call_id and call_id != self._call_id:
            self._call_id, self._turn = call_id, None
            output += self._reset_repeats() + [""] + self._rule(
                f"통화 {call_id}  {clock}", "=", ANSI_BOLD + ANSI_BRIGHT_CYAN
            )

        turn = TURN.search(message)
        if call_id and turn and (call_id, turn.group(1)) != self._turn:
            self._turn = (call_id, turn.group(1))
            output += self._reset_repeats() + [""] + self._rule(
                f"통화 {call_id} · {turn.group(1)}번째 대화  {clock}", "-", ANSI_BOLD
            )

        spoken = SPOKEN.search(message)
        if spoken and "[REALTIME] STT user=" in message:
            output += self._flush_repeats()
            output += self._line(clock, "사용자", spoken.group("text"), ANSI_WHITE, bold=True)
            output.append(" " * 18 + _paint(f"(음성 인식 {int(spoken.group(1)) / 1000:.1f}초)", ANSI_DIM, self.color))
            return output
        if spoken and "[REALTIME] LLM user=" in message:
            output += self._flush_repeats()
            output += self._line(clock, "클론", spoken.group("text"), ANSI_GREEN, bold=True)
            output.append(" " * 18 + _paint(f"(답변 생성 {int(spoken.group(1)) / 1000:.1f}초)", ANSI_DIM, self.color))
            return output
        if "[CALL_TRACE] turn" in message and turn_summary_ko(message):
            done = "turn completed" in message
            status = "대화 완료" if done else "대화 실패" if "turn failed" in message else "대화 건너뜀"
            style = ANSI_GREEN if done else ANSI_RED + ANSI_BOLD
            text = f"{status}: {turn_summary_ko(message)}"
            error = re.search(r"error=(\S+)", message)
            if error:
                text += f"  오류={error.group(1)}"
            return output + self._flush_repeats() + self._line(clock, "턴", text, style, bold=True)

        described = describe(message)
        if described is not None and not described.text:
            return output  # duplicate of another line, hidden on purpose
        if described is None:
            tag, text, style = event_tag(message), compact_event(message), _fallback_style(message)
        else:
            tag, text, style = described.tag, described.text, described.style

        key = (tag, text)
        if key in self._recent:
            self._suppressed += 1
            return output
        output += self._flush_repeats()
        self._recent.append(key)
        output += self._line(clock, tag, text, style)
        if "call closed" in message:
            output += self._rule(f"통화 {call_id or '-'} 끝", "=", ANSI_DIM)
            self._turn = None
            self._recent.clear()
        return output


_ANSI = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Continuously append every call event with full text."
    )
    parser.add_argument(
        "--history-minutes",
        type=int,
        default=60,
        help="show this many minutes of past logs first (0 = live only)",
    )
    parser.add_argument("--scan-lines", type=int, default=3000, help="max past lines")
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
            command=remote_command(args.history_minutes, args.scan_lines),
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
            _paint("MIRROR SOUL - 통화 전체 로그", ANSI_BOLD + ANSI_BRIGHT_CYAN, color),
            "모든 통화의 대화와 처리 과정을 잘리지 않게 이어서 보여 줍니다. 위로 스크롤하면 지난 내용을 볼 수 있습니다.",
            f"최근 {args.history_minutes}분 기록을 먼저 보여 준 뒤 실시간 로그가 이어집니다. 종료: Ctrl+C",
            _paint(f"저장 위치: {save_file.name}", ANSI_DIM, color) if save_file else "파일 저장 안 함 (--no-save)",
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
                text = line.replace("stream disconnected", "서버 연결 끊김").replace(
                    "reconnecting...", "다시 연결 중..."
                )
                emit([_paint(f"{datetime.now():%H:%M:%S} [SYSTEM] {text}", ANSI_RED, color)])
                continue
            emit(formatter.format(line))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.terminate()
        if save_file is not None:
            save_file.close()
    print("\n통화 전체 로그를 종료했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
