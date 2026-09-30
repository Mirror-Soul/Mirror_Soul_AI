from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


USER_PATTERN = re.compile(r"user_uuid=([^\s]+)")
FIELD_PATTERN = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
AI_EVENT_MARKERS = (
    "[RAG_PROFILE]",
    "[VOICE_TRAINING]",
    "[VOICE_TRAINING_QUALITY]",
    "[CLONE_SIMILARITY]",
)
GPU_EVENT_MARKERS = ("[FACE_TRAINING]", "[FACE_SIMILARITY]")

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

STAGE_COLORS = {
    "RAG": ANSI_MAGENTA,
    "VOICE": ANSI_CYAN,
    "FACE": ANSI_BLUE,
}
STATUS_COLORS = {
    "OK": ANSI_GREEN,
    "COMPLETED": ANSI_GREEN,
    "PROCESSING": ANSI_CYAN,
    "WAITING": ANSI_DIM,
    "WARNING": ANSI_YELLOW,
    "STALE": ANSI_YELLOW,
    "FAILED": ANSI_RED + ANSI_BOLD,
    "ERROR": ANSI_RED + ANSI_BOLD,
    "INACTIVE": ANSI_RED + ANSI_BOLD,
    "UNKNOWN": ANSI_YELLOW,
}


@dataclass
class Stage:
    status: str = "WAITING"
    job_id: str = "-"
    clone_id: str = "-"
    score: str = "-"
    detail: str = "No matching event yet"


@dataclass
class PipelineSnapshot:
    user_uuid: str | None
    rag: Stage = field(default_factory=Stage)
    voice: Stage = field(default_factory=Stage)
    face: Stage = field(default_factory=Stage)
    overall_score: str = "-"
    overall_note: str = "No complete score set yet"
    score_components: str = "-"
    data_reliability_score: str = "-"
    penalty_score: str = "-"
    events: list[str] = field(default_factory=list)


@dataclass
class RemoteResult:
    ok: bool
    output: str
    error: str = ""


def _fields(line: str) -> dict[str, str]:
    return dict(FIELD_PATTERN.findall(line))


def _status_from_line(line: str) -> str | None:
    lowered = line.lower()
    if " failed" in lowered or "error=" in lowered:
        return "FAILED"
    if " completed" in lowered or "status=completed" in lowered:
        return "COMPLETED"
    if " processing" in lowered or "preprocessing" in lowered:
        return "PROCESSING"
    return None


def _latest_user(ai_logs: str, gpu_logs: str) -> str | None:
    ai_users = USER_PATTERN.findall(ai_logs)
    if ai_users:
        return ai_users[-1]
    gpu_users = USER_PATTERN.findall(gpu_logs)
    return gpu_users[-1] if gpu_users else None


def _face_block(gpu_lines: list[str], user_uuid: str) -> list[str]:
    starts = [
        index
        for index, line in enumerate(gpu_lines)
        if "[FACE_TRAINING] preprocessing:" in line
        and f"user_uuid={user_uuid}" in line
    ]
    if not starts:
        return [line for line in gpu_lines if f"user_uuid={user_uuid}" in line]

    start = starts[-1]
    end = len(gpu_lines)
    for index in range(start + 1, len(gpu_lines)):
        if "[FACE_TRAINING] preprocessing:" in gpu_lines[index]:
            end = index
            break
    return gpu_lines[start:end]


def parse_pipeline_logs(
    ai_logs: str,
    gpu_logs: str,
    requested_user_uuid: str | None = None,
) -> PipelineSnapshot:
    user_uuid = requested_user_uuid or _latest_user(ai_logs, gpu_logs)
    snapshot = PipelineSnapshot(user_uuid=user_uuid)
    if not user_uuid:
        return snapshot

    ai_lines = ai_logs.splitlines()
    gpu_lines = gpu_logs.splitlines()
    direct_ai = [line for line in ai_lines if f"user_uuid={user_uuid}" in line]
    voice_job_ids = {
        _fields(line)["job_id"]
        for line in direct_ai
        if "[VOICE_TRAINING]" in line and "job_id" in _fields(line)
    }
    related_ai = [
        line
        for line in ai_lines
        if f"user_uuid={user_uuid}" in line
        or (
            (
                "[VOICE_TRAINING]" in line
                or "[VOICE_TRAINING_QUALITY]" in line
            )
            and _fields(line).get("job_id") in voice_job_ids
        )
    ]
    related_face = _face_block(gpu_lines, user_uuid)

    for line in related_ai:
        values = _fields(line)
        if "[RAG_PROFILE]" in line:
            status = _status_from_line(line)
            if status:
                snapshot.rag.status = status
            snapshot.rag.clone_id = values.get("clone_id", snapshot.rag.clone_id)
            if "profile_score" in values:
                snapshot.rag.score = values["profile_score"]
                snapshot.data_reliability_score = values.get(
                    "data_reliability", snapshot.data_reliability_score
                )
                snapshot.penalty_score = values.get(
                    "penalty", snapshot.penalty_score
                )
                snapshot.rag.detail = (
                    f"reliability={values.get('data_reliability', '-')} "
                    f"penalty={values.get('penalty', '-')} "
                    f"callback={values.get('callback_sent', '-')}"
                )
                if values.get("callback_sent", "").lower() == "false":
                    snapshot.rag.status = "WARNING"
            elif status == "FAILED":
                snapshot.rag.detail = line.split("error=", 1)[-1]
            else:
                snapshot.rag.detail = f"samples={values.get('samples', '-')}"

        if "[VOICE_TRAINING]" in line:
            status = _status_from_line(line)
            if status:
                snapshot.voice.status = status
            snapshot.voice.job_id = values.get("job_id", snapshot.voice.job_id)
            snapshot.voice.clone_id = values.get("clone_id", snapshot.voice.clone_id)
            if status == "COMPLETED":
                snapshot.voice.detail = f"voice_id={values.get('voice_id', 'set')}"
            elif status == "FAILED":
                snapshot.voice.detail = line.split("error=", 1)[-1]
            else:
                snapshot.voice.detail = f"files={values.get('files', '-')}"

        if "[VOICE_TRAINING_QUALITY] batch:" in line:
            quality_status = values.get("status", "-").upper()
            if quality_status == "FAILED":
                snapshot.voice.status = "FAILED"
            snapshot.voice.job_id = values.get("job_id", snapshot.voice.job_id)
            snapshot.voice.detail = (
                f"input_quality={quality_status} "
                f"accepted={values.get('accepted', '-')} "
                f"rejected={values.get('rejected', '-')} "
                f"duration={values.get('duration', '-')}"
            )

        if "[CLONE_SIMILARITY] updated:" in line:
            snapshot.overall_score = values.get("overall", "-")
            snapshot.overall_note = "last AI score event"
            snapshot.voice.score = values.get("voice", snapshot.voice.score)
            if snapshot.rag.score == "-":
                snapshot.rag.score = values.get("personality", "-")
            snapshot.data_reliability_score = values.get(
                "data_reliability", snapshot.data_reliability_score
            )
            snapshot.penalty_score = values.get(
                "penalty", snapshot.penalty_score
            )
            snapshot.score_components = (
                f"face={values.get('face', '-')} voice={values.get('voice', '-')} "
                f"profile={values.get('personality', '-')} "
                f"reliability={values.get('data_reliability', '-')} "
                f"penalty={values.get('penalty', '-')}"
            )

    for line in related_face:
        values = _fields(line)
        if "[FACE_TRAINING]" in line:
            status = _status_from_line(line)
            if status:
                snapshot.face.status = status
            snapshot.face.job_id = values.get("job_id", snapshot.face.job_id)
            snapshot.face.clone_id = values.get("clone_id", snapshot.face.clone_id)
            if status == "COMPLETED":
                snapshot.face.score = values.get("face_score", snapshot.face.score)
                snapshot.face.detail = (
                    f"quality={values.get('quality_tier', '-')} "
                    f"profile={values.get('profile', '-')}"
                )
            elif status == "FAILED":
                snapshot.face.detail = line.split("error=", 1)[-1]
            elif "preprocessing" in line:
                snapshot.face.detail = f"files={values.get('files', '-')}"
        if "[FACE_SIMILARITY] completed:" in line:
            snapshot.face.score = values.get("score", snapshot.face.score)

    estimated = _estimate_overall(snapshot)
    if estimated is not None:
        snapshot.overall_score = estimated
        snapshot.overall_note = "estimated from complete AI component logs"
        snapshot.score_components = (
            f"face={snapshot.face.score} voice={snapshot.voice.score} "
            f"profile={snapshot.rag.score} "
            f"reliability={snapshot.data_reliability_score} "
            f"penalty={snapshot.penalty_score}"
        )

    events = [
        line.strip()
        for line in related_ai + related_face
        if any(marker in line for marker in AI_EVENT_MARKERS + GPU_EVENT_MARKERS)
    ]
    snapshot.events = events[-10:]
    return snapshot


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def _estimate_overall(snapshot: PipelineSnapshot) -> str | None:
    values = [
        _decimal(snapshot.face.score),
        _decimal(snapshot.voice.score),
        _decimal(snapshot.rag.score),
        _decimal(snapshot.data_reliability_score),
        _decimal(snapshot.penalty_score),
    ]
    if any(value is None for value in values):
        return None
    face, voice, profile, reliability, penalty = values
    raw = (
        face * Decimal("0.30")
        + voice * Decimal("0.30")
        + profile * Decimal("0.30")
        + reliability * Decimal("0.10")
        - penalty
    )
    final = max(Decimal("0"), min(Decimal("95"), raw * Decimal("0.95")))
    return str(final.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _run_ssh(
    *, host: str, user: str, key: Path, command: str, port: int = 22
) -> RemoteResult:
    args = [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(port),
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


def fetch_ai(args: argparse.Namespace) -> tuple[RemoteResult, dict[str, str], str]:
    since = max(1, args.since_minutes)
    command = (
        "printf '__AI_API__='; systemctl is-active mirrorsoul-ai.service 2>/dev/null || true; "
        "printf '__VOICE_WORKER__='; systemctl is-active mirrorsoul-voice-worker.service 2>/dev/null || true; "
        "echo __LOGS__; "
        f"journalctl -u mirrorsoul-ai.service -u mirrorsoul-voice-worker.service "
        f"--since '-{since} minutes' --no-pager -o cat 2>/dev/null "
        "| grep -E '\\[(RAG_PROFILE|VOICE_TRAINING|CLONE_SIMILARITY)\\]' "
        f"| tail -n {args.lines}"
    )
    result = _run_ssh(
        host=args.ai_host,
        user=args.ai_user,
        key=args.ai_key,
        command=command,
    )
    metadata, logs = _split_remote_output(result.output)
    return result, metadata, logs


def fetch_gpu(args: argparse.Namespace) -> tuple[RemoteResult, dict[str, str], str]:
    command = (
        "printf '__FACE_WORKER__='; "
        "if pgrep -f '[m]odel_training.face_training.worker|[r]un-face-worker.sh' >/dev/null; "
        "then echo active; else echo inactive; fi; "
        "printf '__GPU__='; "
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null || echo unavailable; "
        "echo __LOGS__; "
        f"grep -E '\\[FACE_(TRAINING|SIMILARITY)\\]' {args.face_log} 2>/dev/null "
        f"| tail -n {args.lines}"
    )
    result = _run_ssh(
        host=args.gpu_host,
        user=args.gpu_user,
        key=args.gpu_key,
        port=args.gpu_port,
        command=command,
    )
    metadata, logs = _split_remote_output(result.output)
    return result, metadata, logs


def _service(value: str | None) -> str:
    return "OK" if value == "active" else (value or "UNKNOWN").upper()


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


def _stage_line(name: str, stage: Stage, color: bool = False) -> str:
    name_text = _paint(f"{name:<8}", STAGE_COLORS.get(name, ""), color)
    status_text = _paint(
        f"{stage.status:<10}", _status_color(stage.status), color
    )
    score = stage.score
    if score != "-":
        score = _paint(score, ANSI_BOLD + ANSI_BRIGHT_CYAN, color)
    return (
        f"{name_text} [{status_text}] job={stage.job_id:<5} "
        f"clone={stage.clone_id:<5} score={score}"
    )


def _event_line(line: str, color: bool) -> str:
    lowered = line.lower()
    if (
        " failed" in lowered
        or "error=" in lowered
        or "status=failed" in lowered
        or "status=rejected" in lowered
    ):
        style = STATUS_COLORS["FAILED"]
    elif (
        " completed" in lowered
        or "status=completed" in lowered
        or "status=passed" in lowered
        or "status=accepted" in lowered
    ):
        style = STATUS_COLORS["COMPLETED"]
    elif " processing" in lowered or "preprocessing" in lowered:
        style = STATUS_COLORS["PROCESSING"]
    elif "worker started" in lowered:
        style = ANSI_GREEN
    else:
        style = ANSI_DIM
    return _paint(line, style, color)


def render(
    snapshot: PipelineSnapshot,
    ai_result: RemoteResult,
    ai_meta: dict[str, str],
    gpu_result: RemoteResult,
    gpu_meta: dict[str, str],
    color: bool = False,
    ai_cached: bool = False,
    gpu_cached: bool = False,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ai_connection = (
        "OK"
        if ai_result.ok
        else f"WARNING ({ai_result.error}; showing last data)"
        if ai_cached
        else f"ERROR ({ai_result.error})"
    )
    gpu_connection = (
        "OK"
        if gpu_result.ok
        else f"WARNING ({gpu_result.error}; showing last data)"
        if gpu_cached
        else f"ERROR ({gpu_result.error})"
    )
    gpu_value = gpu_meta.get("GPU", "unknown")
    ai_api = _service(ai_meta.get("AI_API"))
    voice_worker = _service(ai_meta.get("VOICE_WORKER"))
    face_worker = _service(gpu_meta.get("FACE_WORKER"))
    if ai_cached:
        ai_api = f"STALE ({ai_api})"
        voice_worker = f"STALE ({voice_worker})"
    if gpu_cached:
        face_worker = f"STALE ({face_worker})"
    overall_score = snapshot.overall_score
    if overall_score != "-":
        overall_score = _paint(
            overall_score, ANSI_BOLD + ANSI_BRIGHT_CYAN, color
        )
    lines = [
        _paint(
            "MIRROR SOUL - AI PIPELINE MONITOR",
            ANSI_BOLD + ANSI_BRIGHT_CYAN,
            color,
        ),
        f"Updated: {now}",
        "=" * 78,
        f"Target user : {snapshot.user_uuid or 'Waiting for a new AI job...'}",
        "",
        _paint("CONNECTIONS / PROCESSES", ANSI_BOLD, color),
        f"AI server   : {_colored_status(ai_connection, color)}",
        f"AI API      : {_colored_status(ai_api, color)}",
        f"Voice worker: {_colored_status(voice_worker, color)}",
        f"GPU server  : {_colored_status(gpu_connection, color)}",
        f"Face worker : {_colored_status(face_worker, color)}",
        f"GPU         : {gpu_value}  (used MiB, total MiB, utilization %)",
        "",
        _paint("PIPELINE", ANSI_BOLD, color),
        _stage_line("RAG", snapshot.rag, color),
        f"         {snapshot.rag.detail}",
        _stage_line("VOICE", snapshot.voice, color),
        f"         {snapshot.voice.detail}",
        _stage_line("FACE", snapshot.face, color),
        f"         {snapshot.face.detail}",
        "",
        f"{_paint('OVERALL', ANSI_BOLD, color)}  : {overall_score}  "
        f"({snapshot.overall_note})",
        f"COMPONENTS: {snapshot.score_components}",
        "",
        _paint("RECENT AI EVENTS", ANSI_BOLD, color),
    ]
    lines.extend(
        [_event_line(line, color) for line in snapshot.events]
        or [_paint("No matching events yet.", ANSI_DIM, color)]
    )
    lines.extend(["", "Ctrl+C to stop. Logs refresh automatically."])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Show only Mirror Soul AI pipeline health and job progress."
    )
    parser.add_argument("--user-uuid", help="Track one member. Defaults to latest AI job.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize output. Defaults to auto-detecting an interactive terminal.",
    )
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--since-minutes", type=int, default=180)
    parser.add_argument("--lines", type=int, default=800)
    parser.add_argument("--ai-host", default="13.209.220.154")
    parser.add_argument("--ai-user", default="ec2-user")
    parser.add_argument("--ai-key", type=Path, default=repo_root / "mirrorsoul-ai-key.pem")
    parser.add_argument("--gpu-host", default="203.249.75.55")
    parser.add_argument("--gpu-user", default="mirrorsoul")
    parser.add_argument("--gpu-port", type=int, default=40053)
    parser.add_argument(
        "--gpu-key",
        type=Path,
        default=Path.home() / ".ssh" / "mirrorsoul_gpu_vscode_ed25519",
    )
    parser.add_argument(
        "--face-log",
        default="/shareHost/C084003-ai/logs/face-worker.log",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not shutil.which("ssh"):
        print("ERROR: OpenSSH client (ssh) was not found.", file=sys.stderr)
        return 2

    try:
        color = args.color == "always" or (
            args.color == "auto"
            and sys.stdout.isatty()
            and "NO_COLOR" not in os.environ
        )
        last_ai_meta: dict[str, str] = {}
        last_ai_logs = ""
        last_gpu_meta: dict[str, str] = {}
        last_gpu_logs = ""
        while True:
            with ThreadPoolExecutor(max_workers=2) as executor:
                ai_future = executor.submit(fetch_ai, args)
                gpu_future = executor.submit(fetch_gpu, args)
                ai_result, ai_meta, ai_logs = ai_future.result()
                gpu_result, gpu_meta, gpu_logs = gpu_future.result()
            ai_cached = False
            gpu_cached = False
            if ai_result.ok:
                last_ai_meta = ai_meta.copy()
                last_ai_logs = ai_logs
            elif last_ai_meta or last_ai_logs:
                ai_meta = last_ai_meta.copy()
                ai_logs = last_ai_logs
                ai_cached = True
            if gpu_result.ok:
                last_gpu_meta = gpu_meta.copy()
                last_gpu_logs = gpu_logs
            elif last_gpu_meta or last_gpu_logs:
                gpu_meta = last_gpu_meta.copy()
                gpu_logs = last_gpu_logs
                gpu_cached = True
            snapshot = parse_pipeline_logs(ai_logs, gpu_logs, args.user_uuid)
            if not args.once:
                if color:
                    print("\033[H\033[J", end="", flush=True)
                else:
                    os.system("cls" if os.name == "nt" else "clear")
            print(
                render(
                    snapshot,
                    ai_result,
                    ai_meta,
                    gpu_result,
                    gpu_meta,
                    color=color,
                    ai_cached=ai_cached,
                    gpu_cached=gpu_cached,
                )
            )
            if args.once:
                return 0 if ai_result.ok and gpu_result.ok else 1
            time.sleep(max(1.0, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
