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

try:
    from tools.monitor_format import (
        display_width,
        ScreenRefresher,
        enable_windows_ansi,
        wrap_detail,
        compact_event,
        section,
        tagged_event,
        terminal_width,
    )
except ModuleNotFoundError:
    from monitor_format import (
        display_width,
        ScreenRefresher,
        compact_event,
        enable_windows_ansi,
        section,
        tagged_event,
        terminal_width,
        wrap_detail,
    )


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
    detail: str = "아직 기록 없음"


@dataclass
class PipelineSnapshot:
    user_uuid: str | None
    rag: Stage = field(default_factory=Stage)
    voice: Stage = field(default_factory=Stage)
    face: Stage = field(default_factory=Stage)
    overall_score: str = "-"
    overall_note: str = "아직 점수가 모두 나오지 않음"
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
    if " failed" in lowered or "error=" in lowered or "status=failed" in lowered:
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


def recent_users(ai_logs: str, gpu_logs: str, limit: int = 3) -> list[str]:
    """Members whose sign-up started most recently, newest first.

    The AI server log (RAG/voice) is ordered by when each member's training
    started; members only seen in the GPU log are added after them.
    """
    ordered: list[str] = []
    for text in (ai_logs, gpu_logs):
        for user in reversed(USER_PATTERN.findall(text)):
            if user not in ordered:
                ordered.append(user)
    return ordered[: max(1, limit)]


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
                if "documents" in values:
                    snapshot.rag.detail += (
                        f" docs={values['documents']}"
                        f" removed={values.get('removed', '0')}"
                    )
                if values.get("callback_sent", "").lower() == "false":
                    snapshot.rag.status = "WARNING"
            elif status == "FAILED":
                snapshot.rag.detail = line.split("error=", 1)[-1]
            else:
                snapshot.rag.detail = f"samples={values.get('samples', '-')}"

        if "[VOICE_TRAINING] status published:" in line:
            # Result-queue status events only confirm the state change; keep
            # the detail from the processing/completed line.
            status = _status_from_line(line)
            if status:
                snapshot.voice.status = status
            snapshot.voice.job_id = values.get("job_id", snapshot.voice.job_id)
        elif "[VOICE_TRAINING]" in line:
            status = _status_from_line(line)
            if status:
                snapshot.voice.status = status
            snapshot.voice.job_id = values.get("job_id", snapshot.voice.job_id)
            snapshot.voice.clone_id = values.get("clone_id", snapshot.voice.clone_id)
            if status == "COMPLETED":
                # Since the result-queue worker, the voice score is logged on
                # the completion line; the overall score is computed by the
                # backend and no longer logged by the AI server.
                snapshot.voice.score = values.get("voice_score", snapshot.voice.score)
                snapshot.voice.detail = (
                    f"voice_score={values.get('voice_score', '-')} "
                    f"voice_id={values.get('voice_id', 'set')}"
                )
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
            snapshot.overall_note = "마지막 점수 로그 기준"
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
        snapshot.overall_note = "AI 로그로 계산한 예상값"
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
    if value == "active":
        return "OK"
    if value in {"activating", "reloading"}:
        # systemd shows "activating" while a crashing service is restarting.
        return "WARNING (재시작 반복 중 - journalctl 확인)"
    return (value or "UNKNOWN").upper()


def _kv(label: str, value: str, width: int = 14) -> str:
    """``label : value`` with the label padded by display width (Korean-safe)."""
    return f"{label}{' ' * max(0, width - display_width(label))}: {value}"


def _detail_ko(name: str, stage: Stage) -> str:
    """Show a stage's raw ``key=value`` detail as a short Korean sentence."""
    detail = stage.detail
    values = dict(FIELD_PATTERN.findall(detail))
    if stage.status == "FAILED" and not values:
        return f"실패: {detail}"
    if name == "RAG":
        if "reliability" in values:
            parts = [
                f"데이터 신뢰도 {values['reliability']}",
                f"감점 {values.get('penalty', '-')}",
            ]
            if "docs" in values:
                parts.append(f"저장 문서 {values['docs']}개 (정리 {values.get('removed', '0')}개)")
            callback = values.get("callback", "-").lower()
            parts.append(
                "백엔드 전달 완료" if callback == "true"
                else "백엔드 전달 실패" if callback == "false"
                else "백엔드 전달 확인 안 됨"
            )
            return " · ".join(parts)
        if "samples" in values:
            return f"인터뷰 답변 {values['samples']}개로 학습 중"
    if name == "VOICE":
        if "voice_score" in values:
            return f"음성 점수 {values['voice_score']} · 음성 ID {values.get('voice_id', '있음')}"
        if "input_quality" in values:
            result = {"PASSED": "통과", "FAILED": "실패"}.get(values["input_quality"], values["input_quality"])
            return (
                f"음성 샘플 품질 {result} · 사용 {values.get('accepted', '-')}개 · "
                f"제외 {values.get('rejected', '-')}개"
            )
        if "files" in values:
            return f"음성 파일 {values['files']}개로 학습 중"
    if name == "FACE":
        if "quality" in values:
            job = re.search(r"job-(\d+)", detail)
            saved = f"얼굴 프로필 저장됨 (job {job.group(1)})" if job else "얼굴 프로필 저장됨"
            return f"품질 등급 {values['quality']} · {saved}"
        if "files" in values:
            return f"얼굴 영상 {values['files']}개 처리 중"
    if detail == "아직 기록 없음":
        return "아직 시작 안 됨"
    return detail


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


STAGE_KO = {"RAG": "성격·기억", "VOICE": "음성", "FACE": "얼굴"}


def _stage_line(name: str, stage: Stage, color: bool = False) -> str:
    name_text = _paint(f"{name:<6}", STAGE_COLORS.get(name, ""), color)
    korean = STAGE_KO.get(name, "")
    korean_text = korean + " " * max(0, 10 - display_width(korean))
    status_text = _paint(
        f"{stage.status:<10}", _status_color(stage.status), color
    )
    score = stage.score
    if score != "-":
        score = _paint(score, ANSI_BOLD + ANSI_BRIGHT_CYAN, color)
    return (
        f"{name_text}{korean_text}[{status_text}] 점수 {score}"
        + (f"   (job {stage.job_id})" if stage.job_id != "-" else "")
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


def _event_style(line: str) -> str:
    lowered = line.lower()
    if (
        " failed" in lowered
        or "error=" in lowered
        or "status=failed" in lowered
        or "status=rejected" in lowered
    ):
        return STATUS_COLORS["FAILED"]
    if (
        " completed" in lowered
        or "status=completed" in lowered
        or "status=passed" in lowered
        or "status=accepted" in lowered
    ):
        return STATUS_COLORS["COMPLETED"]
    if " processing" in lowered or "preprocessing" in lowered:
        return STATUS_COLORS["PROCESSING"]
    if "skipped" in lowered or "bypassed" in lowered or "warning" in lowered:
        return ANSI_YELLOW
    if "worker started" in lowered:
        return ANSI_GREEN
    return ANSI_DIM


def _detail_lines(detail: str, width: int, color: bool) -> str:
    parts = wrap_detail(detail, width, 11)
    lines = [f"         └ {parts[0]}", *(" " * 11 + part for part in parts[1:])]
    return "\n".join(_paint(line, ANSI_DIM, color) for line in lines)


def _components_ko(snapshot: PipelineSnapshot) -> str:
    return (
        f"얼굴 {snapshot.face.score} · 음성 {snapshot.voice.score} · "
        f"성격 {snapshot.rag.score} · 신뢰도 {snapshot.data_reliability_score} · "
        f"감점 {snapshot.penalty_score}"
    )


def _member_block(
    snapshot: PipelineSnapshot,
    *,
    index: int,
    total: int,
    width: int,
    color: bool,
    event_limit: int,
) -> list[str]:
    clone = next(
        (
            stage.clone_id
            for stage in (snapshot.rag, snapshot.face, snapshot.voice)
            if stage.clone_id not in {"-", "None"}
        ),
        "-",
    )
    user = snapshot.user_uuid or "-"
    order = "가장 최근" if index == 0 else f"{index + 1}번째 최근"
    title = f"회원 {index + 1}/{total} ({order}): {user[:8]}… · clone {clone}"
    overall = snapshot.overall_score
    if overall != "-":
        overall = _paint(overall, ANSI_BOLD + ANSI_BRIGHT_CYAN, color)
    lines = [
        section(title, color, width),
        _stage_line("RAG", snapshot.rag, color),
        _detail_lines(_detail_ko("RAG", snapshot.rag), width, color),
        _stage_line("VOICE", snapshot.voice, color),
        _detail_lines(_detail_ko("VOICE", snapshot.voice), width, color),
        _stage_line("FACE", snapshot.face, color),
        _detail_lines(_detail_ko("FACE", snapshot.face), width, color),
        _kv("예상 종합점수", f"{overall}  ({snapshot.overall_note})"),
        _kv("구성 점수", _components_ko(snapshot)),
    ]
    events = [
        text
        for text in (
            _tagged_event_line(line, color, width)
            for line in snapshot.events[-event_limit:]
        )
        if text
    ]
    if events:
        lines.append(_paint("최근 이벤트 (위가 오래된 것)", ANSI_DIM, color))
        lines.extend(events)
    return lines


def _tagged_event_line(line: str, color: bool, width: int | None = None) -> str:
    return tagged_event(line, _event_style(line), color=color, width=width)


def render(
    snapshot: PipelineSnapshot,
    ai_result: RemoteResult,
    ai_meta: dict[str, str],
    gpu_result: RemoteResult,
    gpu_meta: dict[str, str],
    color: bool = False,
    ai_cached: bool = False,
    gpu_cached: bool = False,
    members: list[PipelineSnapshot] | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ai_connection = (
        "OK"
        if ai_result.ok
        else f"WARNING ({ai_result.error}; 마지막으로 받은 데이터 표시)"
        if ai_cached
        else f"ERROR ({ai_result.error})"
    )
    gpu_connection = (
        "OK"
        if gpu_result.ok
        else f"WARNING ({gpu_result.error}; 마지막으로 받은 데이터 표시)"
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
    members = members or [snapshot]
    width = terminal_width()
    target = (
        f"최근 가입한 회원 {len(members)}명 (최신 순)"
        if snapshot.user_uuid
        else "새 학습 작업을 기다리는 중..."
    )
    lines = [
        _paint(
            "MIRROR SOUL - AI 학습 모니터  (회원가입: RAG 성격·기억 / 음성 / 얼굴)",
            ANSI_BOLD + ANSI_BRIGHT_CYAN,
            color,
        ),
        f"Updated: {now}",
        "=" * min(width, 100),
        _kv("표시 대상", _paint(target, ANSI_BOLD, color)),
        "",
        section("연결 상태 (CONNECTIONS / PROCESSES)", color, width),
        _kv("AI 서버", _colored_status(ai_connection, color)),
        _kv("AI API", _colored_status(ai_api, color)),
        _kv("음성 워커", _colored_status(voice_worker, color)),
        _kv("GPU 서버", _colored_status(gpu_connection, color)),
        _kv("얼굴 워커", _colored_status(face_worker, color)),
        _kv("GPU 사용량", f"{gpu_value}  (사용 MiB, 전체 MiB, 사용률 %)"),
    ]
    if not snapshot.user_uuid:
        lines += ["", _paint("아직 학습 기록이 없습니다.", ANSI_DIM, color)]
    else:
        for index, member in enumerate(members):
            lines.append("")
            lines.extend(
                _member_block(
                    member,
                    index=index,
                    total=len(members),
                    width=width,
                    color=color,
                    event_limit=6 if index == 0 else 3,
                )
            )
    lines.extend([
        "",
        _paint(
            "예상 종합점수는 AI 로그로 계산한 값이고, 공식 클론 점수는 백엔드가 계산해서 저장합니다.\n"
            "색상: 초록=완료  청록=진행 중  노랑=주의  빨강=실패  회색=대기"
            "   |  종료: Ctrl+C\n"
            "전체 학습 기록 보기: tools\\monitor-ai-events.cmd --color always --gpu-port 40053",
            ANSI_DIM,
            color,
        ),
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Show only Mirror Soul AI pipeline health and job progress."
    )
    parser.add_argument("--user-uuid", help="Track one member. Defaults to the most recent members.")
    parser.add_argument(
        "--members",
        type=int,
        default=3,
        help="how many recent members to show (default 3)",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize output. Defaults to auto-detecting an interactive terminal.",
    )
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--since-minutes", type=int, default=180)
    parser.add_argument("--lines", type=int, default=2000)
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

    enable_windows_ansi()
    try:
        color = args.color == "always" or (
            args.color == "auto"
            and sys.stdout.isatty()
            and "NO_COLOR" not in os.environ
        )
        screen = ScreenRefresher(color=color)
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
            if args.user_uuid:
                users = [args.user_uuid]
            else:
                users = recent_users(ai_logs, gpu_logs, args.members)
            members = [
                parse_pipeline_logs(ai_logs, gpu_logs, user) for user in users
            ] or [parse_pipeline_logs(ai_logs, gpu_logs, args.user_uuid)]
            snapshot = members[0]
            frame = render(
                snapshot,
                ai_result,
                ai_meta,
                gpu_result,
                gpu_meta,
                color=color,
                ai_cached=ai_cached,
                gpu_cached=gpu_cached,
                members=members,
            )
            if args.once:
                print(frame)
            else:
                screen.show(frame)
            if args.once:
                return 0 if ai_result.ok and gpu_result.ok else 1
            time.sleep(max(1.0, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
