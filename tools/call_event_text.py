"""Plain-Korean descriptions of call-server log lines for the monitors.

Each known event becomes a short Korean sentence plus its key numbers.
Technical names (Ditto, WebRTC, ICE, RAG, STT, LLM, TTS) stay as they are.
Unknown lines return ``None`` so callers can show the original text.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"

OK = ANSI_GREEN
INFO = ANSI_DIM
RUN = ANSI_CYAN
WARN = ANSI_YELLOW
FAIL = ANSI_RED + ANSI_BOLD

INPUT_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz mono PCM16 utterances

_FIELD = re.compile(r"\b([A-Za-z_]+)=([^\s,]+)")
_SPEECH = re.compile(r"elapsed_ms=\d+:\s?(.*)$")

STATE_KO = {
    "new": "준비",
    "checking": "확인 중",
    "connecting": "연결 중",
    "connected": "연결됨",
    "completed": "완료",
    "failed": "실패",
    "disconnected": "끊김",
    "closed": "종료",
    "gathering": "수집 중",
    "complete": "완료",
}
SOURCE_KO = {
    "profile_snapshot": "프로필",
    "interview_memory": "인터뷰",
    "conversation_memory": "대화 기억",
    "preference_memory": "선호",
    "member_profile_summary": "프로필(이전 형식)",
    "member_profile_interview": "인터뷰(이전 형식)",
    "interview_answer": "인터뷰(이전 형식)",
}
SPEAKER_KO = {
    "USER": "사용자",
    "CLONE": "클론",
}
CLOSE_STATUS_KO = {
    "COMPLETED": "정상 종료",
    "COMPLETED_WITH_ERRORS": "일부 오류 후 종료",
    "FAILED": "실패",
}
CLOSE_REASON_KO = {
    "CALL_END": "사용자가 종료",
    "PEER_CLOSED": "앱 연결이 끊김",
    "SESSION_CLOSED": "세션 정리",
}


@dataclass(frozen=True)
class EventText:
    tag: str
    text: str
    style: str = INFO


def _fields(message: str) -> dict[str, str]:
    return dict(_FIELD.findall(message))


def _seconds(ms: str | None) -> str:
    try:
        return f"{int(ms or 0) / 1000:.1f}초"
    except ValueError:
        return "-"


def _duration(ms: str | None) -> str:
    try:
        total = int(ms or 0) // 1000
    except ValueError:
        return "-"
    minutes, seconds = divmod(total, 60)
    return f"{minutes}분 {seconds}초" if minutes else f"{seconds}초"


def _short_user(value: str | None) -> str:
    return f"{value[:8]}…" if value and len(value) > 8 else (value or "-")


def _media(value: str | None) -> str:
    return {"VIDEO": "영상 통화", "VOICE": "음성 통화"}.get(value or "", value or "-")


def _sources(value: str | None) -> str:
    if not value or value == "none":
        return "없음"
    parts = []
    for item in value.split(","):
        name, _, count = item.partition(":")
        parts.append(f"{SOURCE_KO.get(name, name)} {count or '1'}")
    return ", ".join(parts)


def _received_type(message: str) -> str | None:
    match = re.search(r"'type':\s*'([A-Z_]+)'", message)
    return match.group(1) if match else None


def _ice_kind(message: str) -> str:
    match = re.search(r"candidate:\S+ \d+ (udp|tcp) \d+ \S+ \d+ typ (\w+)", message)
    return f"{match.group(1)} {match.group(2)}" if match else "후보"


def describe(message: str) -> EventText | None:
    """Return a Korean description of one call-server log message."""
    f = _fields(message)
    low = message.lower()

    # --- signaling --------------------------------------------------------
    if message.startswith("[SIGNALING]"):
        if "disconnected:" in message:
            code = re.search(r"HTTP (\d{3})", message)
            reason = f"HTTP {code.group(1)}" if code else message.split("disconnected:", 1)[1].strip()
            return EventText("SIGNAL", f"시그널링 서버 연결 끊김 ({reason})", FAIL)
        if "reconnecting in" in message:
            return EventText("SIGNAL", "잠시 후 다시 연결합니다", WARN)
        if "connecting..." in message:
            return EventText("SIGNAL", "시그널링 서버에 연결 중", RUN)
        if message.rstrip().endswith("connected"):
            return EventText("SIGNAL", "시그널링 서버 연결됨", OK)
        if "JOIN sent" in message:
            return EventText("SIGNAL", "AI 서버 입장 요청 보냄", INFO)
        if "CALL_ACCEPT sent" in message:
            return EventText(
                "SIGNAL",
                f"통화 수락: {_media(f.get('mediaType'))}, 클론 회원 "
                f"{_short_user(f.get('user'))} (clone {f.get('clone_id', '-')})",
                OK,
            )
        if "CALL_REJECT sent" in message:
            return EventText("SIGNAL", f"통화 거절: {f.get('reason', '-')}", FAIL)
        if "ANSWER sent" in message:
            return EventText("SIGNAL", "연결 응답(SDP answer) 보냄", INFO)
        if "CALL_END handled" in message:
            return EventText("SIGNAL", "통화 종료 처리 완료", INFO)
        kind = _received_type(message)
        if kind == "JOINED":
            return EventText("SIGNAL", "시그널링 입장 완료, 통화 요청 대기 중", OK)
        if kind == "CALL_INVITE":
            return EventText("SIGNAL", "통화 요청 받음", RUN)
        if kind == "OFFER":
            return EventText("SIGNAL", "앱의 연결 정보(SDP offer) 받음", INFO)
        if kind == "ICE":
            return EventText("SIGNAL", f"앱의 네트워크 후보(ICE) 받음: {_ice_kind(message)}", INFO)
        if kind == "CALL_END":
            return EventText("SIGNAL", "앱에서 통화 종료 요청", RUN)
        if kind:
            return EventText("SIGNAL", f"{kind} 메시지 받음", INFO)
        return None

    # --- WebRTC -----------------------------------------------------------
    if message.startswith("[WEBRTC]"):
        if "RTC configuration created" in message:
            turn = "TURN 사용" if f.get("turn") != "disabled" else "TURN 미사용"
            return EventText("WEBRTC", f"WebRTC 설정 생성 ({turn})", INFO)
        if "peer connection created" in message:
            return EventText("WEBRTC", "WebRTC 연결 생성", INFO)
        if "output audio track added" in message:
            return EventText("WEBRTC", "클론 음성 트랙 추가", INFO)
        if "output video track added" in message:
            renderer = "Ditto 사용" if f.get("renderer") == "enabled" else "Ditto 꺼짐"
            style = INFO if f.get("renderer") == "enabled" else WARN
            return EventText("VIDEO", f"클론 영상 트랙 추가 ({renderer})", style)
        if "session created" in message:
            return EventText("WEBRTC", "통화 세션 생성", INFO)
        if "applying remote offer" in message:
            return EventText("WEBRTC", "앱 연결 정보 적용", INFO)
        if "track received" in message:
            kind = {"audio": "음성", "video": "영상"}.get(f.get("kind", ""), f.get("kind", ""))
            return EventText("WEBRTC", f"사용자 {kind} 트랙 받음", INFO)
        if "starting realtime pipeline" in message:
            return EventText("WEBRTC", "실시간 대화 처리 시작", OK)
        if "realtime pipeline attached" in message:
            return EventText("WEBRTC", "실시간 대화 처리 연결됨", INFO)
        if "ICE gathering" in message:
            return EventText("WEBRTC", f"ICE 후보 수집: {STATE_KO.get(f.get('state', ''), f.get('state', '-'))}", INFO)
        if "local answer created" in message:
            return EventText("WEBRTC", "연결 응답 생성", INFO)
        if "remote ICE added" in message:
            return EventText("WEBRTC", "네트워크 후보 추가", INFO)
        if "ICE connection" in message:
            state = f.get("state", "-")
            style = OK if state in {"completed", "connected"} else FAIL if state == "failed" else INFO
            return EventText("WEBRTC", f"ICE 연결: {STATE_KO.get(state, state)}", style)
        if "connection:" in message and "state=" in message:
            state = f.get("state", "-")
            style = OK if state == "connected" else FAIL if state == "failed" else INFO
            return EventText("WEBRTC", f"WebRTC 연결: {STATE_KO.get(state, state)}", style)
        if "idle portrait ready" in message:
            return EventText("VIDEO", "대기 사진 준비 완료", OK)
        if "idle loop preparation failed" in message:
            return EventText("VIDEO", "Ditto 대기 영상 준비 실패 (정지 사진으로 진행)", WARN)
        if "idle portrait preparation failed" in message:
            return EventText("VIDEO", "대기 사진 준비 실패", FAIL)
        if "Ditto video renderer unavailable" in message:
            return EventText("VIDEO", "Ditto 렌더러를 사용할 수 없음", FAIL)
        return None

    # --- Ditto ------------------------------------------------------------
    if message.startswith("[DITTO_CALL]"):
        if "face profile loaded" in message:
            job = re.search(r"job-(\d+)", message)
            return EventText("VIDEO", f"얼굴 프로필 불러옴 (얼굴 학습 job {job.group(1) if job else '-'})", INFO)
        if "profile ready for call" in message:
            return EventText("VIDEO", "Ditto 얼굴 준비 완료", OK)
        if "idle loop render started" in message:
            return EventText("VIDEO", f"Ditto 대기 영상 렌더 시작 ({f.get('seconds', '-')}초 분량)", RUN)
        if "idle loop ready" in message:
            source = "저장된 영상 재사용" if f.get("source") == "cache" else "새로 렌더"
            return EventText("VIDEO", f"Ditto 대기 영상 준비 완료 ({source}, {_seconds(f.get('elapsed_ms'))})", OK)
        if "idle loop unavailable" in message or "idle loop rejected" in message:
            return EventText("VIDEO", f"Ditto 대기 영상 실패, 정지 사진으로 진행 ({f.get('error_code', '-')})", WARN)
        if "render slot acquired" in message:
            workers = "워커 2개 모두 사용" if f.get("reserved_slots") == "2" else f"워커 {f.get('worker', '-')}"
            return EventText(
                "VIDEO",
                f"GPU 배정: {workers}, 답변 길이 {float(f.get('audio_seconds', 0)):.1f}초, "
                f"대기 {_seconds(f.get('queue_wait_ms'))}",
                INFO,
            )
        if "render queued behind another call" in message:
            return EventText("VIDEO", "다른 통화의 렌더가 끝나길 기다리는 중", WARN)
        if "video integrity passed" in message:
            return EventText(
                "VIDEO",
                f"영상 검증 통과: {f.get('frames', '-')}프레임, "
                f"{float(f.get('duration', '0s').rstrip('s')):.1f}초, {f.get('resolution', '-')}",
                INFO,
            )
        if "render completed" in message:
            return EventText("VIDEO", f"Ditto 렌더 완료 ({float(f.get('seconds', 0)):.1f}초 걸림)", OK)
        if "retrying render" in message:
            return EventText("VIDEO", f"Ditto 렌더 재시도 ({f.get('attempt', '-')}번째)", WARN)
        return None

    # --- video output -----------------------------------------------------
    if message.startswith("[VIDEO_OUT]"):
        if "crossfade started" in message:
            names = {"idle-loop": "대기 영상", "idle-portrait": "대기 사진", "reply": "답변"}
            return EventText(
                "VIDEO",
                f"화면 전환: {names.get(f.get('from', ''), f.get('from', '-'))} → "
                f"{names.get(f.get('to', ''), f.get('to', '-'))}",
                INFO,
            )
        if "segment completed" in message:
            return EventText("VIDEO", "답변 영상 재생 끝", INFO)
        if "decode failed" in message:
            return EventText("VIDEO", "답변 영상 재생 오류, 대기 화면으로 돌아감", FAIL)
        if "idle loop video failed" in message:
            return EventText("VIDEO", "대기 영상 재생 오류, 정지 사진으로 전환", WARN)
        # queued/idle-ready lines repeat the [DITTO_CALL]/[REALTIME] ones.
        return EventText("", "", "")

    # --- realtime pipeline ------------------------------------------------
    if message.startswith("[REALTIME]"):
        if "utterance enqueued" in message:
            try:
                seconds = max(0, int(f.get("wav_bytes", "0")) - 44) / INPUT_BYTES_PER_SECOND
                length = f"약 {seconds:.1f}초"
            except ValueError:
                length = "-"
            return EventText("CALL", f"사용자 말 감지 ({length})", RUN)
        if "utterance dequeued" in message:
            return EventText("CALL", "이번 턴 처리 시작", INFO)
        if "utterance dropped" in message or "queue full" in low:
            return EventText("CALL", "처리 중이라 사용자 말을 건너뜀", WARN)
        if "STT start" in message:
            return EventText("STT", "음성 인식 시작", RUN)
        if "context ready" in message:
            voice = "있음" if f.get("voice_id") == "set" else "없음"
            return EventText("CALL", f"클론 정보 준비 (MBTI {f.get('mbti', '-')}, 음성 ID {voice})", INFO)
        if "RAG lookup complete" in message:
            count = f.get("count", "0")
            style = WARN if count == "0" else OK
            distance = f.get("best_distance", "none")
            distance_text = "" if distance == "none" else f", 가장 가까운 거리 {float(distance):.2f}"
            mode = "AI 서버 저장소" if f.get("mode") == "remote" else "로컬 저장소(주의)"
            sources = re.search(r"\bsources=(\S+)", message)
            failed = f.get("failed_queries", "0")
            failed_text = f", 검색 {failed}건 실패" if failed != "0" else ""
            return EventText(
                "RAG",
                f"기억 검색 {count}건: {_sources(sources.group(1) if sources else None)} "
                f"({_seconds(f.get('elapsed_ms'))}, {mode}{distance_text}{failed_text})",
                WARN if failed != "0" else style,
            )
        if "RAG lookup skipped" in message:
            return EventText("RAG", f"기억 검색 건너뜀 ({f.get('error_code', '-')}), 기억 없이 답변", WARN)
        if "LLM start" in message:
            return EventText("LLM", "답변 생성 시작", RUN)
        if "TTS start" in message:
            return EventText("TTS", "음성 합성 시작", RUN)
        if "TTS complete" in message:
            return EventText("TTS", f"음성 합성 완료 ({_seconds(f.get('elapsed_ms'))})", OK)
        if "queueing reply audio" in message:
            return EventText("CALL", "답변 음성 준비됨", INFO)
        if "Ditto reply render started" in message:
            return EventText("VIDEO", "Ditto 답변 영상 렌더 시작", RUN)
        if "Ditto reply video queued" in message:
            return EventText("VIDEO", f"답변 영상 준비 완료 (영상 단계 {_seconds(f.get('elapsed_ms'))})", OK)
        if "Ditto reply render failed" in message:
            return EventText("VIDEO", f"Ditto 답변 영상 실패 ({f.get('error_code', '-')})", FAIL)
        if "reply audio suppressed" in message:
            return EventText("CALL", "영상이 준비되지 않아 답변을 내보내지 않음", FAIL)
        if "reply audio queued" in message:
            return EventText("CALL", "답변 재생 시작", OK)
        if "conversation context updated" in message:
            return EventText("CALL", f"대화 맥락 갱신 (지금까지 {f.get('turns', '-')}턴)", INFO)
        if "incoming audio track ended" in message:
            return EventText("CALL", "사용자 음성 트랙 끝남", INFO)
        if "starting realtime audio tasks" in message:
            return EventText("CALL", "실시간 음성 처리 시작", INFO)
        if "realtime audio tasks started" in message:
            return EventText("", "", "")
        if "pipeline failed" in message:
            return EventText("CALL", "대화 처리 실패: " + message.split("failed:", 1)[-1].strip(), FAIL)
        return None

    # --- call history (talk logs) ----------------------------------------
    if message.startswith("[TALK_LOG]"):
        who = SPEAKER_KO.get(f.get("speaker", ""), f.get("speaker", "발화"))
        if "saved:" in message:
            if f.get("duplicated") == "yes":
                return EventText("기록", f"{who} 발화는 이미 저장됨 (중복 요청)", INFO)
            return EventText(
                "기록",
                f"통화 기록 저장: {who} 발화 {f.get('chars', '-')}자 "
                f"(기록 번호 {f.get('talkLogId', '-')})",
                OK,
            )
        if "save failed" in message:
            return EventText(
                "기록",
                f"통화 기록 저장 실패: {who} 발화, 오류 {f.get('error_code', '-')}",
                FAIL,
            )
        if "saving disabled" in message:
            reason = message.split("reason=", 1)[-1].strip() if "reason=" in message else "-"
            return EventText("기록", f"통화 기록 저장 꺼짐 ({reason})", WARN)
        if "entry dropped after close" in message:
            return EventText("기록", f"통화 종료 후 도착한 {who} 발화는 저장하지 않음", WARN)
        if "turn not recorded" in message:
            return EventText("기록", "이번 턴을 통화 기록에 넣지 못함", WARN)
        if "call summary" in message:
            failed = f.get("failed", "0")
            dropped = f.get("dropped", "0")
            style = OK if failed == "0" and dropped == "0" else WARN
            return EventText(
                "기록",
                f"통화 기록 정리: 저장 {f.get('saved', '-')}건, "
                f"중복 {f.get('duplicated', '-')}건, 실패 {failed}건, 누락 {dropped}건",
                style,
            )
        return None

    # --- trace ------------------------------------------------------------
    if message.startswith("[CALL_TRACE]"):
        if "in-flight tasks cancelled" in message:
            return EventText("TRACE", f"진행 중이던 작업 정리 ({f.get('count', '-')}개)", INFO)
        if "call closed" in message:
            status = f.get("status", "-")
            style = OK if status == "COMPLETED" else WARN if status == "COMPLETED_WITH_ERRORS" else FAIL
            text = (
                f"통화 종료: {CLOSE_STATUS_KO.get(status, status)}, "
                f"{CLOSE_REASON_KO.get(f.get('reason', ''), f.get('reason', '-'))}, "
                f"통화 시간 {_duration(f.get('duration_ms'))}, "
                f"턴 {f.get('turns_completed', '-')}/{f.get('turns_started', '-')} 완료"
            )
            if f.get("turns_failed", "0") != "0":
                text += f", 실패 {f['turns_failed']}"
            if f.get("last_error", "none") != "none":
                text += f", 마지막 오류 {f['last_error']}"
            return EventText("TRACE", text, style)
        return None
    return None


def speech_text(message: str) -> str | None:
    match = _SPEECH.search(message)
    return match.group(1) if match else None


TIMING_KO = (
    ("stt", "음성 인식"),
    ("rag", "기억 검색"),
    ("llm", "답변 생성"),
    ("tts", "음성 합성"),
    ("video", "영상"),
)


def turn_summary_ko(message: str) -> str | None:
    values = {name: int(value) for name, value in re.findall(r"\b(\w+)_ms=(\d+)", message)}
    if "total" not in values:
        return None
    parts = [f"총 {values['total'] / 1000:.1f}초"]
    parts += [
        f"{label} {values[key] / 1000:.1f}"
        for key, label in TIMING_KO
        if key in values
    ]
    return " | ".join(parts)
