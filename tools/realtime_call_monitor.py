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
MEDIA_PATTERN = re.compile(r"mediaType['\"]?\s*[:=]\s*['\"]?([^,'\"\s}]+)")


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
        if "call_reject sent" in lowered:
            snapshot.signal = Stage("FAILED", line)

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
    return "OK" if value == "active" else (value or "UNKNOWN").upper()


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
    return f"{label} [{status}] {stage.detail}"


def _event_line(line: str, color: bool) -> str:
    lowered = line.lower()
    if any(word in lowered for word in ("failed", "error=", "reject", "unavailable")):
        style = ANSI_RED + ANSI_BOLD
    elif any(word in lowered for word in ("completed", "connected", "ready", "queued", "accept")):
        style = ANSI_GREEN
    elif any(word in lowered for word in ("start", "processing", "received", "created")):
        style = ANSI_CYAN
    elif "warning" in lowered or "skipped" in lowered or "retry" in lowered:
        style = ANSI_YELLOW
    else:
        style = ANSI_DIM
    return _paint(line, style, color)


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
    engine = ditto_ready.get("engine") if isinstance(ditto_ready.get("engine"), dict) else {}

    server_connection = (
        "OK"
        if remote.ok
        else f"WARNING ({remote.error}; showing last data)"
        if cached
        else f"ERROR ({remote.error})"
    )
    api_status = "OK" if call_health.get("status") == "ok" else "ERROR"
    ditto_status = "READY" if ditto_ready.get("status") == "ready" else "ERROR"
    call_service = _service(metadata.get("CALL_SERVICE"))
    tunnel_service = _service(metadata.get("TUNNEL_SERVICE"))
    signaling = snapshot.signaling_connection
    if cached:
        call_service = f"STALE ({call_service})"
        api_status = f"STALE ({api_status})"
        signaling = f"STALE ({signaling})"
        tunnel_service = f"STALE ({tunnel_service})"
        ditto_status = f"STALE ({ditto_status})"
    busy = engine.get("busy")
    busy_text = "YES" if busy is True else "NO" if busy is False else "UNKNOWN"
    busy_style = ANSI_YELLOW if busy is True else ANSI_GREEN if busy is False else ANSI_YELLOW
    last_render = engine.get("lastRenderSeconds")
    last_render_text = f"{last_render}s" if last_render is not None else "-"
    last_error = engine.get("lastError") or "none"
    error_style = ANSI_GREEN if last_error == "none" else ANSI_RED + ANSI_BOLD

    lines = [
        _paint(
            "MIRROR SOUL - REALTIME CALL MONITOR",
            ANSI_BOLD + ANSI_BRIGHT_CYAN,
            color,
        ),
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        _paint("CONNECTIONS / SERVICES", ANSI_BOLD, color),
        f"Call server  : {_colored_status(server_connection, color)}",
        f"Call service : {_colored_status(call_service, color)}",
        f"Call API     : {_colored_status(api_status, color)}",
        f"Signaling    : {_colored_status(signaling, color)}",
        f"Ditto tunnel : {_colored_status(tunnel_service, color)}",
        f"Ditto GPU    : {_colored_status(ditto_status, color)}",
        f"GPU model    : {engine.get('gpu', '-')}",
        f"GPU busy     : {_paint(busy_text, busy_style, color)}",
        f"Render count : {engine.get('renderCount', '-')}  "
        f"last={last_render_text}",
        f"Last error   : {_paint(str(last_error), error_style, color)}",
        "",
        _paint("LATEST CALL", ANSI_BOLD, color),
        f"Call ID      : {snapshot.call_id}",
        f"Clone user   : {snapshot.user_uuid}",
        f"Media type   : {snapshot.media_type}",
        "",
        _paint("CALL PIPELINE", ANSI_BOLD, color),
        _stage_line("SIGNAL", snapshot.signal, color),
        _stage_line("WEBRTC", snapshot.webrtc, color),
        _stage_line("STT", snapshot.stt, color),
        _stage_line("RAG", snapshot.rag, color),
        _stage_line("LLM", snapshot.llm, color),
        _stage_line("TTS", snapshot.tts, color),
        _stage_line("VIDEO", snapshot.video, color),
        "",
        _paint("LATEST TURN SUMMARY", ANSI_BOLD, color),
        _event_line(snapshot.trace_summary, color),
        "",
        _paint("RECENT CLOSED CALLS", ANSI_BOLD, color),
    ]
    lines.extend(
        [_event_line(line, color) for line in snapshot.recent_call_summaries]
        or [_paint("No closed call summary yet", ANSI_DIM, color)]
    )
    lines.extend([
        "",
        _paint("RECENT CALL EVENTS", ANSI_BOLD, color),
    ])
    lines.extend(
        [_event_line(line, color) for line in snapshot.events]
        or [_paint("Waiting for a new call...", ANSI_DIM, color)]
    )
    lines.extend(["", "Ctrl+C to stop. Status refreshes automatically."])
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
