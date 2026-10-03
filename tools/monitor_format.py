"""Shared helpers that keep monitor event lines short, tagged and readable."""

from __future__ import annotations

import re
import shutil

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


def tagged_event(
    line: str,
    message_style: str,
    *,
    color: bool,
    width: int | None = None,
) -> str:
    tag = event_tag(line)
    label = paint(f"[{tag:<6}]", ANSI_BOLD + TAG_STYLES.get(tag, ""), color)
    available = None if width is None else max(20, width - 9)
    return f"{label} {paint(compact_event(line, available), message_style, color)}"


def section(title: str, color: bool, width: int) -> str:
    bar = "-" * max(0, min(width, 100) - len(title) - 4)
    return paint(f"-- {title} {bar}", ANSI_BOLD, color)
