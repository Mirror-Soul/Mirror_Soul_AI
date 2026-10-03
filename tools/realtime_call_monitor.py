from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    from tools.monitor_format import (
        compact_event,
        section,
        tagged_event,
        terminal_width,
    )
except ModuleNotFoundError:
    from monitor_format import compact_event, section, tagged_event, terminal_width


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_BRIGHT_CYAN = "\033[96m"

STATUS_COLORS = {
    "OK": ANSI_GREEN,
    "READY": ANSI_GREEN,
    "CONNECTED": ANSI_GREEN,
    "COMPLETED": ANSI_GREEN,
    "ENDED": ANSI_GREEN,
    "PROCESSING": ANSI_CYAN,
    "CONNECTING": ANSI_CYAN,
    "WAITING": ANSI_DIM,
    "SKIPPED": ANSI_DIM,
    "WARNING": ANSI_YELLOW,
    "STALE": ANSI_YELLOW,
    "RECONNECTING": ANSI_YELLOW,
    "FAILED": ANSI_RED + ANSI_BOLD,
    "ERROR": ANSI_RED + ANSI_BOLD,
    "INACTIVE": ANSI_RED + ANSI_BOLD,
    "UNKNOWN": ANSI_YELLOW,
}

CALL_MARKERS = (
    "[SIGNALING]",
    "[WEBRTC]",
    "[REALTIME]",
    "[DITTO_CALL]",
    "[CALL_TRACE]",
)
CALL_ID_PATTERN = re.compile(r"callId['\"]?\s*[:=]\s*['\"]?([^,'\"\s}]+)")
USER_PATTERN = re.compile(
    r"(?:cloneUserUuid['\"]?\s*[:=]\s*['\"]?|user=)([^,'\"\s}]+)"
)
MEDIA_PATTERN = re.compile(
    r"(?:mediaType['\"]?\s*[:=]\s*['\"]?|media_type=)([^,'\"\s}]+)"
)
REASON_PATTERN = re.compile(r"reason['\"]?\s*[:=]\s*['\"]?([^,'\"\s}]+)")


@dataclass
class Stage:
    status: str = "WAITING"
    detail: str = "No matching event yet"


@dataclass
class CallSnapshot:
    signaling_connection: str = "UNKNOWN"
    call_id: str = "-"
    user_uuid: str = "-"
    media_type: str = "-"
    signal: Stage = field(default_factory=Stage)
    webrtc: Stage = field(default_factory=Stage)
    stt: Stage = field(default_factory=Stage)
    rag: Stage = field(default_factory=Stage)
    llm: Stage = field(default_factory=Stage)
    tts: Stage = field(default_factory=Stage)
    video: Stage = field(default_factory=Stage)
    trace_summary: str = "No completed turn yet"
    recent_call_summaries: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class RemoteResult:
    ok: bool
    output: str
    error: str = ""


def _paint(text: str, style: str, enabled: bool) -> str:
    return f"{style}{text}{ANSI_RESET}" if enabled and style else text


def _status_color(value: str) -> str:
    upper = value.upper()
    if upper.startswith("ERROR"):
        return STATUS_COLORS["ERROR"]
    if upper.startswith("WARNING") or upper.startswith("STALE"):
        return ANSI_YELLOW
    if upper.startswith("OK"):
        return ANSI_GREEN
    return STATUS_COLORS.get(upper, "")


def _colored_status(value: str, enabled: bool) -> str:
    return _paint(value, _status_color(value), enabled)


def _last_value(pattern: re.Pattern[str], lines: list[str]) -> str:
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return match.group(1)
    return "-"


def _call_block(lines: list[str]) -> list[str]:
    invite_starts = [
        index
        for index, line in enumerate(lines)
        if "'type': 'CALL_INVITE'" in line
        or '"type": "CALL_INVITE"' in line
    ]
    if invite_starts:
        return lines[invite_starts[-1] :]
    starts = [
        index
        for index, line in enumerate(lines)
        if "[SIGNALING] CALL_ACCEPT sent:" in line
        or "[SIGNALING] CALL_REJECT sent:" in line
    ]
    return lines[starts[-1] :] if starts else []


def parse_call_logs(logs: str) -> CallSnapshot:
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    snapshot = CallSnapshot()

    for line in lines:
        if "[SIGNALING] connected" in line and "disconnected" not in line:
            snapshot.signaling_connection = "CONNECTED"
        elif "[SIGNALING] connecting" in line:
            snapshot.signaling_connection = "CONNECTING"
        elif "[SIGNALING] disconnected" in line:
            snapshot.signaling_connection = "RECONNECTING"

    block = _call_block(lines)
    snapshot.recent_call_summaries = [
        line for line in lines if "[CALL_TRACE] call closed:" in line
    ][-5:]
    if not block:
        snapshot.events = [line for line in lines if line.startswith(CALL_MARKERS)][-12:]
        return snapshot

    identity_lines = [
        line
        for line in block
        if "'type': 'CALL_INVITE'" in line
        or '"type": "CALL_INVITE"' in line
        or "[SIGNALING] CALL_ACCEPT sent:" in line
        or "[SIGNALING] CALL_REJECT sent:" in line
    ]
    snapshot.call_id = _last_value(CALL_ID_PATTERN, identity_lines or block)
    if snapshot.call_id != "-":
        block = [
            line
            for line in block
            if (match := CALL_ID_PATTERN.search(line)) is None
            or match.group(1) == snapshot.call_id
        ]
    snapshot.user_uuid = _last_value(USER_PATTERN, block)
    snapshot.media_type = _last_value(MEDIA_PATTERN, block).upper()

    if snapshot.media_type == "VOICE":
        snapshot.video = Stage("SKIPPED", "Voice-only call")

    for line in block:
        lowered = line.lower()

        if "'type': 'call_invite'" in lowered or '"type": "call_invite"' in lowered:
            snapshot.signal = Stage("PROCESSING", "Call invite received")
        if "call_accept sent" in lowered:
            snapshot.signal = Stage("COMPLETED", "Call accepted")
        if "call_accept sent" in lowered and "mediatype=" in lowered:
            snapshot.signal = Stage(
                "COMPLETED", "Call accepted (context loaded from backend API)"
            )
        if "call_reject sent" in lowered:
            reason = REASON_PATTERN.search(line)
            snapshot.signal = Stage(
                "FAILED",
                f"Call rejected: {reason.group(1) if reason else line}",
            )

        if "[webrtc] peer connection created" in lowered:
            snapshot.webrtc = Stage("PROCESSING", "Peer connection created")
        if "[webrtc] connection: connected" in lowered or (
            "[webrtc] connection:" in lowered and "state=connected" in lowered
        ):
            snapshot.webrtc = Stage("CONNECTED", "WebRTC connected")
        if "[webrtc] connection: failed" in lowered or (
            "[webrtc] connection:" in lowered and "state=failed" in lowered
        ):
            snapshot.webrtc = Stage("FAILED", line)
        if "[signaling] call_end handled" in lowered:
            snapshot.webrtc = Stage("ENDED", "Call ended normally")

        if "[realtime] stt start" in lowered:
            snapshot.stt = Stage("PROCESSING", "Speech recognition started")
        if "[realtime] stt user=" in lowered:
            snapshot.stt = Stage("COMPLETED", line.split(":", 1)[-1].strip())
        if "[realtime] stt empty result" in lowered:
            snapshot.stt = Stage("WARNING", "No speech recognized")

        if "[realtime] rag lookup complete" in lowered:
            if " count=0 " in f"{lowered} ":
                snapshot.rag = Stage(
                    "WARNING",
                    "No member memories found (count=0) - check RAG search mode/data",
                )
            else:
                snapshot.rag = Stage("COMPLETED", line)
        if "[realtime] rag lookup skipped" in lowered:
            snapshot.rag = Stage("WARNING", line)

        if "[realtime] llm start" in lowered:
            snapshot.llm = Stage("PROCESSING", "Reply generation started")
        if "[realtime] llm user=" in lowered:
            snapshot.llm = Stage("COMPLETED", line.split(":", 1)[-1].strip())

        if "[realtime] tts start" in lowered:
            snapshot.tts = Stage("PROCESSING", "Voice synthesis started")
        if "[realtime] tts complete" in lowered:
            snapshot.tts = Stage("COMPLETED", line)

        if "renderer=enabled" in lowered or "face profile loaded" in lowered:
            snapshot.video = Stage("PROCESSING", "Ditto profile readying")
        if "idle portrait ready" in lowered:
            snapshot.video = Stage("READY", "Idle portrait ready")
        if "ditto reply render started" in lowered:
            snapshot.video = Stage("PROCESSING", "Reply video rendering")
        if "[ditto_call] render completed" in lowered:
            snapshot.video = Stage("COMPLETED", line)
        if "ditto reply video queued" in lowered:
            snapshot.video = Stage("COMPLETED", "Reply video queued")
        if (
            "ditto reply render failed" in lowered
            or "ditto video renderer unavailable" in lowered
            or "idle portrait preparation failed" in lowered
        ):
            snapshot.video = Stage("FAILED", line)

        if "[realtime] pipeline failed" in lowered:
            detail = line.split("failed:", 1)[-1].strip()
            for stage in (snapshot.stt, snapshot.rag, snapshot.llm, snapshot.tts):
                if stage.status == "PROCESSING":
                    stage.status = "FAILED"
                    stage.detail = detail

        if "[call_trace] turn completed:" in lowered:
            snapshot.trace_summary = line
        if "[call_trace] turn failed:" in lowered:
            snapshot.trace_summary = line
        if "[call_trace] turn skipped:" in lowered:
            snapshot.trace_summary = line

    snapshot.events = [line for line in block if line.startswith(CALL_MARKERS)][-20:]
    return snapshot


def _run_ssh(*, host: str, user: str, key: Path, command: str) -> RemoteResult:
    args = [
        "ssh",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        f"{user}@{host}",
        command,
    ]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RemoteResult(False, "", "SSH query timed out")
    except OSError as exc:
        return RemoteResult(False, "", str(exc))
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "SSH failed").strip()
        return RemoteResult(False, result.stdout, error.splitlines()[-1])
    return RemoteResult(True, result.stdout)


def _split_remote_output(output: str) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    logs: list[str] = []
    in_logs = False
    for line in output.splitlines():
        if line == "__LOGS__":
            in_logs = True
            continue
        if not in_logs and line.startswith("__") and "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip("_")] = value.strip()
        elif in_logs:
            logs.append(line)
    return metadata, "\n".join(logs)


def fetch_call_server(
    args: argparse.Namespace,
) -> tuple[RemoteResult, dict[str, str], str]:
    since = max(1, args.since_minutes)
    command = (
        "printf '__CALL_SERVICE__='; "
        "systemctl is-active mirror-soul-call.service 2>/dev/null || true; "
        "printf '__TUNNEL_SERVICE__='; "
        "systemctl is-active mirror-soul-ditto-tunnel.service 2>/dev/null || true; "
        "printf '__CALL_HEALTH__='; "
        "curl -fsS --max-time 4 http://127.0.0.1:8000/health 2>/dev/null || echo unavailable; "
        "echo; printf '__DITTO_READY__='; "
        "curl -fsS --max-time 4 http://127.0.0.1:18080/ready 2>/dev/null || echo unavailable; "
        "echo; if systemctl cat mirror-soul-ditto-tunnel-2.service >/dev/null 2>&1; then "
        "printf '__TUNNEL_SERVICE_2__='; "
        "systemctl is-active mirror-soul-ditto-tunnel-2.service 2>/dev/null || true; "
        "printf '__DITTO_READY_2__='; "
        "curl -fsS --max-time 4 http://127.0.0.1:18081/ready 2>/dev/null || echo unavailable; "
        "echo; fi; "
        "echo; echo __LOGS__; "
        "journalctl -u mirror-soul-call.service "
        f"--since '-{since} minutes' -n {args.lines} --no-pager -o cat 2>/dev/null"
    )
    result = _run_ssh(
        host=args.call_host,
        user=args.call_user,
        key=args.call_key,
        command=command,
    )
    metadata, logs = _split_remote_output(result.output)
    return result, metadata, logs


def _service(value: str | None) -> str:
    if value == "active":
        return "OK"
    if value in {"activating", "reloading"}:
        return "WARNING (restarting - check journalctl)"
    return (value or "UNKNOWN").upper()


def _json_object(value: str | None) -> dict:
    if not value or value == "unavailable":
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stage_line(name: str, stage: Stage, color: bool) -> str:
    label = _paint(f"{name:<8}", ANSI_BLUE, color)
    status = _paint(f"{stage.status:<10}", _status_color(stage.status), color)
    detail = compact_event(stage.detail, max(20, terminal_width() - 23))
    return f"{label} [{status}] {detail}"


def _event_style(line: str) -> str:
    lowered = line.lower()
    if any(word in lowered for word in ("failed", "error=", "reject", "unavailable")):
        return ANSI_RED + ANSI_BOLD
    if "warning" in lowered or "skipped" in lowered or "retry" in lowered:
        return ANSI_YELLOW
    if any(word in lowered for word in ("completed", "connected", "ready", "queued", "accept")):
        return ANSI_GREEN
    if any(word in lowered for word in ("start", "processing", "received", "created")):
        return ANSI_CYAN
    return ANSI_DIM


def _event_line(line: str, color: bool) -> str:
    return _paint(line, _event_style(line), color)


def _tagged_event_line(line: str, color: bool, width: int | None = None) -> str:
    return tagged_event(line, _event_style(line), color=color, width=width)


def render(
    snapshot: CallSnapshot,
    remote: RemoteResult,
    metadata: dict[str, str],
    *,
    color: bool = False,
    cached: bool = False,
) -> str:
    call_health = _json_object(metadata.get("CALL_HEALTH"))
    ditto_ready = _json_object(metadata.get("DITTO_READY"))
    ditto_ready_2 = _json_object(metadata.get("DITTO_READY_2"))
    worker_states = [ditto_ready]
    if metadata.get("DITTO_READY_2"):
        worker_states.append(ditto_ready_2)
    engines = [
        state.get("engine")
        for state in worker_states
        if isinstance(state.get("engine"), dict)
    ]
    engine = engines[0] if engines else {}

    server_connection = (
        "OK"
        if remote.ok
        else f"WARNING ({remote.error}; showing last data)"
        if cached
        else f"ERROR ({remote.error})"
    )
    api_status = "OK" if call_health.get("status") == "ok" else "ERROR"
    ready_workers = sum(state.get("status") == "ready" for state in worker_states)
    worker_count = len(worker_states)
    ditto_status = "READY" if ready_workers == worker_count else "ERROR"
    call_service = _service(metadata.get("CALL_SERVICE"))
    tunnel_service = _service(metadata.get("TUNNEL_SERVICE"))
    if metadata.get("TUNNEL_SERVICE_2"):
        tunnel_2 = _service(metadata.get("TUNNEL_SERVICE_2"))
        tunnel_service = "OK" if tunnel_service == tunnel_2 == "OK" else "ERROR"
    signaling = snapshot.signaling_connection
    rag_mode = str(call_health.get("ragSearch") or "")
    rag_search = (
        "OK (AI server store)"
        if rag_mode == "remote"
        else "WARNING (local store - set RAG_SEARCH_BASE_URL)"
        if rag_mode == "local"
        else "UNKNOWN"
    )
    if cached:
        call_service = f"STALE ({call_service})"
        api_status = f"STALE ({api_status})"
        signaling = f"STALE ({signaling})"
        tunnel_service = f"STALE ({tunnel_service})"
        ditto_status = f"STALE ({ditto_status})"
    busy_values = [item.get("busy") for item in engines]
    busy = True if True in busy_values else False if busy_values else None
    busy_text = "YES" if busy is True else "NO" if busy is False else "UNKNOWN"
    busy_style = ANSI_YELLOW if busy is True else ANSI_GREEN if busy is False else ANSI_YELLOW
    last_renders = [
        item.get("lastRenderSeconds")
        for item in engines
        if item.get("lastRenderSeconds") is not None
    ]
    last_render = max(last_renders) if last_renders else None
    last_render_text = f"{last_render}s" if last_render is not None else "-"
    errors = [item.get("lastError") for item in engines if item.get("lastError")]
    last_error = "; ".join(str(error) for error in errors) or "none"
    render_count = sum(
        int(item.get("renderCount") or 0)
        for item in engines
    )
    error_style = ANSI_GREEN if last_error == "none" else ANSI_RED + ANSI_BOLD

    width = terminal_width()
    media_style = ANSI_BOLD + (
        ANSI_BLUE if snapshot.media_type == "VIDEO" else ANSI_CYAN
    )
    lines = [
        _paint(
            "MIRROR SOUL - REALTIME CALL MONITOR  (signaling / STT / RAG / LLM / TTS / video)",
            ANSI_BOLD + ANSI_BRIGHT_CYAN,
            color,
        ),
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * min(width, 100),
        "",
        section("CONNECTIONS / SERVICES", color, width),
        f"Call server  : {_colored_status(server_connection, color)}",
        f"Call service : {_colored_status(call_service, color)}",
        f"Call API     : {_colored_status(api_status, color)}",
        f"Signaling    : {_colored_status(signaling, color)}",
        f"RAG search   : {_colored_status(rag_search, color)}",
        f"Ditto tunnel : {_colored_status(tunnel_service, color)}",
        f"Ditto GPU    : {_colored_status(ditto_status, color)}",
        f"Ditto workers: {ready_workers}/{worker_count} READY",
        f"GPU model    : {engine.get('gpu', '-')}",
        f"GPU busy     : {_paint(busy_text, busy_style, color)}",
        f"Render count : {render_count}  "
        f"last={last_render_text}",
        f"Last error   : {_paint(str(last_error), error_style, color)}",
        "",
        section("LATEST CALL", color, width),
        f"Call ID      : {_paint(snapshot.call_id, ANSI_BOLD, color)}",
        f"Clone user   : {snapshot.user_uuid}",
        f"Media type   : {_paint(snapshot.media_type, media_style, color)}",
        "",
        section("CALL PIPELINE (latest turn)", color, width),
        _stage_line("SIGNAL", snapshot.signal, color),
        _stage_line("WEBRTC", snapshot.webrtc, color),
        _stage_line("STT", snapshot.stt, color),
        _stage_line("RAG", snapshot.rag, color),
        _stage_line("LLM", snapshot.llm, color),
        _stage_line("TTS", snapshot.tts, color),
        _stage_line("VIDEO", snapshot.video, color),
        "",
        section("LATEST TURN SUMMARY", color, width),
        _tagged_event_line(snapshot.trace_summary, color, width)
        if snapshot.trace_summary.startswith("[")
        else _paint(snapshot.trace_summary, ANSI_DIM, color),
        "",
        section("RECENT CLOSED CALLS", color, width),
    ]
    lines.extend(
        [
            _tagged_event_line(line, color, width)
            for line in snapshot.recent_call_summaries
        ]
        or [_paint("No closed call summary yet", ANSI_DIM, color)]
    )
    lines.extend([
        "",
        section("RECENT CALL EVENTS (oldest -> newest)", color, width),
    ])
    lines.extend(
        [_tagged_event_line(line, color, width) for line in snapshot.events]
        or [_paint("Waiting for a new call...", ANSI_DIM, color)]
    )
    lines.extend([
        "",
        _paint(
            "Colors: green=done  cyan=running  yellow=warning  red=failed  gray=waiting"
            "   |  Ctrl+C to stop",
            ANSI_DIM,
            color,
        ),
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Show Mirror Soul realtime AI call health and progress."
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--since-minutes", type=int, default=180)
    parser.add_argument("--lines", type=int, default=1800)
    parser.add_argument("--call-host", default="43.202.181.134")
    parser.add_argument("--call-user", default="ec2-user")
    parser.add_argument(
        "--call-key",
        type=Path,
        default=repo_root / "mirrorsoul-call-key.pem",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not shutil.which("ssh"):
        print("ERROR: OpenSSH client (ssh) was not found.", file=sys.stderr)
        return 2

    color = args.color == "always" or (
        args.color == "auto"
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
    )
    try:
        last_metadata: dict[str, str] = {}
        last_logs = ""
        while True:
            remote, metadata, logs = fetch_call_server(args)
            cached = False
            if remote.ok:
                last_metadata = metadata.copy()
                last_logs = logs
            elif last_metadata or last_logs:
                metadata = last_metadata.copy()
                logs = last_logs
                cached = True
            snapshot = parse_call_logs(logs)
            if not args.once:
                if color:
                    print("\033[H\033[J", end="", flush=True)
                else:
                    os.system("cls" if os.name == "nt" else "clear")
            print(render(snapshot, remote, metadata, color=color, cached=cached))
            if args.once:
                return 0 if remote.ok else 1
            time.sleep(max(1.0, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
