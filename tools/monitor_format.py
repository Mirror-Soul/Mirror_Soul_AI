"""Shared helpers that keep monitor event lines short, tagged and readable."""

from __future__ import annotations

import os
import re
import shutil
import unicodedata

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"
ANSI_BRIGHT_CYAN = "\033[96m"

TAG_STYLES = {
    "RAG": ANSI_MAGENTA,
    "VOICE": ANSI_CYAN,
    "FACE": ANSI_BLUE,
    "SCORE": ANSI_BRIGHT_CYAN,
    "SIGNAL": ANSI_MAGENTA,
    "WEBRTC": ANSI_BLUE,
    "STT": ANSI_CYAN,
    "LLM": ANSI_CYAN,
    "TTS": ANSI_CYAN,
    "VIDEO": ANSI_BLUE,
    "TRACE": ANSI_WHITE,
    "CALL": ANSI_WHITE,
}

_MARKER = re.compile(r"^\s*\[([A-Z_]+)\]\s*")
_UUID = re.compile(
    r"\b([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_LONG_PATH = re.compile(r"(?:s3://|/)[^\s,'\"]{40,}")


def terminal_width(default: int = 110) -> int:
    try:
        return max(60, shutil.get_terminal_size((default, 30)).columns)
    except (OSError, ValueError):
        return default


def paint(text: str, style: str, enabled: bool) -> str:
    return f"{style}{text}{ANSI_RESET}" if enabled and style else text


def event_tag(line: str) -> str:
    """Map a raw log line to the pipeline stage it belongs to."""
    match = _MARKER.match(line)
    marker = match.group(1) if match else ""
    lowered = line.lower()
    if marker == "RAG_PROFILE":
        return "RAG"
    if marker.startswith("VOICE_TRAINING"):
        return "VOICE"
    if marker.startswith("FACE_"):
        return "FACE"
    if marker == "CLONE_SIMILARITY":
        return "SCORE"
    if marker == "SIGNALING":
        return "SIGNAL"
    if marker == "WEBRTC":
        return "VIDEO" if "portrait" in lowered or "video track" in lowered else "WEBRTC"
    if marker == "DITTO_CALL":
        return "VIDEO"
    if marker == "CALL_TRACE":
        return "TRACE"
    if marker == "REALTIME":
        for keyword, tag in (
            ("stt", "STT"),
            ("rag ", "RAG"),
            ("llm", "LLM"),
            ("tts", "TTS"),
            ("ditto", "VIDEO"),
            ("video", "VIDEO"),
        ):
            if keyword in lowered:
                return tag
        return "CALL"
    return marker[:6] or "LOG"


def _short_path(match: re.Match[str]) -> str:
    value = match.group(0)
    prefix = "s3://" if value.startswith("s3://") else ""
    parts = [part for part in value[len(prefix):].split("/") if part]
    if len(parts) <= 2:
        return value
    return f"{prefix}…/{'/'.join(parts[-2:])}"


def compact_event(line: str, width: int | None = None) -> str:
    """Drop the raw marker, shorten UUIDs/paths and fit the line to width."""
    text = _MARKER.sub("", line.strip(), count=1)
    text = _UUID.sub(lambda m: f"{m.group(1)}…", text)
    text = _LONG_PATH.sub(_short_path, text)
    if width is not None and width > 0 and len(text) > width:
        text = text[: max(1, width - 1)].rstrip() + "…"
    return text


def display_width(text: str) -> int:
    """Terminal columns used by text: Korean and other wide glyphs take two."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap on spaces by display width; never drop characters."""
    width = max(10, width)
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if display_width(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while display_width(word) > width:
            piece = ""
            for char in word:
                if display_width(piece + char) > width:
                    break
                piece += char
            lines.append(piece)
            word = word[len(piece):]
        current = word
    lines.append(current)
    return lines


_TIMING = re.compile(r"\b(total|stt|context|rag|llm|tts|video)_ms=(\d+)")
_TIMING_LABELS = (
    ("stt", "STT"),
    ("rag", "RAG"),
    ("llm", "LLM"),
    ("tts", "TTS"),
    ("video", "VIDEO"),
)


def timing_summary(line: str) -> str | None:
    """'total_ms=16690 stt_ms=1032 ...' -> 'total 16.7s | STT 1.0s | ...'."""
    values = {name: int(value) for name, value in _TIMING.findall(line)}
    if "total" not in values:
        return None
    parts = [f"total {values['total'] / 1000:.1f}s"]
    parts.extend(
        f"{label} {values[key] / 1000:.1f}s"
        for key, label in _TIMING_LABELS
        if key in values
    )
    return " | ".join(parts)


def pretty_timings(text: str) -> str:
    """Replace raw ``*_ms=`` fields with one readable seconds summary."""
    summary = timing_summary(text)
    if summary is None:
        return text
    stripped = re.sub(r"\s*\b(total|stt|context|rag|llm|tts|video)_ms=\d+", "", text)
    return f"{stripped.rstrip()}  -> {summary}"


def tagged_event(
    line: str,
    message_style: str,
    *,
    color: bool,
    width: int | None = None,
    prefix: str = "",
    prefix_width: int = 0,
) -> str:
    """Tag label + full message, wrapped under the message column.

    Nothing is cut off: long questions and answers continue on the next
    lines, aligned after the tag so the tag column stays readable.
    """
    tag = event_tag(line)
    label = paint(f"[{tag:<6}]", ANSI_BOLD + TAG_STYLES.get(tag, ""), color)
    text = pretty_timings(compact_event(line))
    indent = prefix_width + 9
    if width is None:
        wrapped = [text]
    else:
        wrapped = wrap_text(text, width - indent - 1)
    body = paint(wrapped[0], message_style, color)
    rest = [" " * indent + paint(part, message_style, color) for part in wrapped[1:]]
    return "\n".join([f"{prefix}{label} {body}", *rest])


def wrap_detail(text: str, width: int, indent: int) -> list[str]:
    """Wrap a stage detail so its continuation lines start at ``indent``."""
    return wrap_text(compact_event(text), max(20, width - indent - 1))


def section(title: str, color: bool, width: int) -> str:
    bar = "-" * max(0, min(width, 100) - len(title) - 4)
    return paint(f"-- {title} {bar}", ANSI_BOLD, color)


def enable_windows_ansi() -> None:
    """Let the classic Windows console (CMD) interpret ANSI colors."""
    if os.name == "nt":
        os.system("")


class ScreenRefresher:
    """Redraw a dashboard without stacking copies in the scrollback.

    The previous ``ESC[H ESC[J`` only cleared the visible screen, so a frame
    taller than the window left older copies above it and the view looked
    like it was overwritten mid-way. This clears the scrollback as well and
    skips redraws when nothing but the clock changed, so the screen stays
    still while you read it.
    """

    def __init__(self, *, color: bool) -> None:
        self.color = color
        self._last_key: str | None = None

    def show(self, frame: str, *, ignore_prefix: str = "Updated:") -> bool:
        key = "\n".join(
            line for line in frame.splitlines() if not line.startswith(ignore_prefix)
        )
        if key == self._last_key:
            return False
        self._last_key = key
        if self.color:
            print("\033[H\033[2J\033[3J", end="", flush=True)
        else:
            os.system("cls" if os.name == "nt" else "clear")
        print(frame, flush=True)
        return True
