"""Plain-Korean descriptions of sign-up training logs (RAG, voice, face).

Same idea as ``call_event_text``: known events become short Korean
sentences with their key numbers; unknown lines return ``None`` so the
monitor shows the original text.
"""

from __future__ import annotations

import re

try:
    from tools.call_event_text import FAIL, INFO, OK, RUN, WARN, EventText
except ModuleNotFoundError:
    from call_event_text import FAIL, INFO, OK, RUN, WARN, EventText

_FIELD = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")
STATUS_KO = {
    "PROCESSING": "처리 중",
    "COMPLETED": "완료",
    "FAILED": "실패",
    "PASSED": "통과",
}


def _fields(message: str) -> dict[str, str]:
    return dict(_FIELD.findall(message))


def _member(f: dict[str, str]) -> str:
    user = f.get("user_uuid", "")
    clone = f.get("clone_id")
    text = f"회원 {user[:8]}…" if user else ""
    if clone and clone != "None":
        text += f" (clone {clone})" if text else f"clone {clone}"
    return text


def _with_member(text: str, f: dict[str, str]) -> str:
    member = _member(f)
    return f"{text} · {member}" if member else text


def describe(message: str) -> EventText | None:
    f = _fields(message)

    if message.startswith("[RAG_PROFILE]"):
        if "processing:" in message:
            return EventText("RAG", _with_member(f"성격·기억(RAG) 학습 시작: 인터뷰 답변 {f.get('samples', '-')}개", f), RUN)
        if "completed:" in message:
            return EventText(
                "RAG",
                _with_member(
                    f"성격·기억(RAG) 학습 완료: 성격 점수 {f.get('profile_score', '-')}, "
                    f"데이터 신뢰도 {f.get('data_reliability', '-')}, 감점 {f.get('penalty', '0')}, "
                    f"저장 문서 {f.get('documents', '-')}개 (정리 {f.get('removed', '0')}개)",
                    f,
                ),
                OK,
            )
        if "failed:" in message:
            return EventText("RAG", _with_member(f"성격·기억(RAG) 학습 실패: {message.split('error=', 1)[-1]}", f), FAIL)
        return None

    if message.startswith("[VOICE_TRAINING_QUALITY]"):
        if "batch:" in message:
            passed = f.get("status") == "PASSED"
            return EventText(
                "VOICE",
                f"음성 샘플 품질 검사 {'통과' if passed else '실패'}: "
                f"사용 {f.get('accepted', '-')}개, 제외 {f.get('rejected', '-')}개 (job {f.get('job_id', '-')})",
                OK if passed else FAIL,
            )
        if "sample:" in message and "REJECTED" in message:
            return EventText(
                "VOICE",
                f"음성 샘플 {f.get('sample', '-')}번 제외: {f.get('reasons', '-')} (job {f.get('job_id', '-')})",
                WARN,
            )
        if "sample:" in message:
            return EventText(
                "VOICE",
                f"음성 샘플 {f.get('sample', '-')}번 사용: 길이 {f.get('duration', '-')}, "
                f"음량 {f.get('rms_dbfs', '-')}dBFS, 무음 비율 {f.get('silence_ratio', '-')}",
                INFO,
            )
        if "bypassed:" in message:
            return EventText("VOICE", f"음성 샘플 품질 검사 생략 (job {f.get('job_id', '-')})", WARN)
        return None

    if message.startswith("[VOICE_TRAINING]"):
        if "processing:" in message:
            return EventText("VOICE", _with_member(f"음성 학습 시작: 음성 파일 {f.get('files', '-')}개 (job {f.get('job_id', '-')})", f), RUN)
        if "completed:" in message:
            return EventText("VOICE", f"음성 학습 완료: 음성 점수 {f.get('voice_score', '-')} (job {f.get('job_id', '-')})", OK)
        if "status published:" in message:
            status = f.get("status", "-")
            style = OK if status == "COMPLETED" else FAIL if status == "FAILED" else INFO
            return EventText("VOICE", f"백엔드에 음성 학습 상태 전달: {STATUS_KO.get(status, status)} (job {f.get('job_id', '-')})", style)
        if "job failed:" in message:
            return EventText("VOICE", f"음성 학습 실패 (job {f.get('job_id', '-')}): {message.split('error=', 1)[-1]}", FAIL)
        if "reusing completed result" in message:
            return EventText("VOICE", f"이미 끝난 음성 학습 결과 재사용 (job {f.get('job_id', '-')})", INFO)
        if "worker started" in message:
            return EventText("VOICE", "음성 학습 워커 시작", OK)
        if "no message available" in message:
            return EventText("", "", "")
        if "publish failed" in message:
            return EventText("VOICE", f"백엔드에 상태 전달 실패, 요청 보관 (job {f.get('job_id', '-')})", FAIL)
        return None

    if message.startswith("[CLONE_SIMILARITY]"):
        if "speaker embedding result" in message:
            return EventText(
                "SCORE",
                f"목소리 유사도 측정: {f.get('score', '-')}점 (코사인 {f.get('cosine', '-')}, 원본 샘플 {f.get('samples', '-')}개)",
                OK,
            )
        if "reference audio saved" in message:
            return EventText("SCORE", "목소리 비교용 클론 음성 생성 완료", INFO)
        if "skipped" in message:
            return EventText("SCORE", "목소리 유사도 측정 생략 (기본 점수 사용)", WARN)
        if "failed" in message:
            return EventText("SCORE", "목소리 유사도 측정 실패 (기본 점수 사용)", WARN)
        return None

    if message.startswith("[FACE_TRAINING]"):
        if "preprocessing:" in message:
            return EventText("FACE", _with_member(f"얼굴 학습 시작: 영상 {f.get('files', '-')}개 (job {f.get('job_id', '-')})", f), RUN)
        if "video preprocessed:" in message:
            return EventText(
                "FACE",
                f"얼굴 영상 전처리 완료: 길이 {f.get('duration', '-')}, 해상도 {f.get('resolution', '-')}, "
                f"추출 프레임 {f.get('frames', '-')}장",
                INFO,
            )
        if "frame analysis completed:" in message:
            gate = f.get("quality_gate") == "True"
            return EventText(
                "FACE",
                f"얼굴 프레임 분석: 사용 {f.get('accepted', '-')}장, 제외 {f.get('rejected', '-')}장, "
                f"품질 기준 {'통과' if gate else '미달'}",
                OK if gate else WARN,
            )
        if message.startswith("[FACE_TRAINING] completed:"):
            return EventText(
                "FACE",
                _with_member(
                    f"얼굴 학습 완료: 얼굴 점수 {f.get('face_score', '-')}, 품질 등급 {f.get('quality_tier', '-')} "
                    f"(job {f.get('job_id', '-')})",
                    f,
                ),
                OK,
            )
        if "status published:" in message:
            status = f.get("status", "-")
            style = OK if status == "COMPLETED" else FAIL if status == "FAILED" else INFO
            return EventText("FACE", f"백엔드에 얼굴 학습 상태 전달: {STATUS_KO.get(status, status)} (job {f.get('job_id', '-')})", style)
        if "job failed:" in message:
            return EventText("FACE", f"얼굴 학습 실패 (job {f.get('job_id', '-')}): {message.split('error=', 1)[-1]}", FAIL)
        if "request message deleted" in message:
            return EventText("FACE", "처리한 얼굴 학습 요청 정리", INFO)
        if "request visibility extended" in message:
            return EventText("", "", "")
        if "no message available" in message:
            return EventText("", "", "")
        if "worker started" in message:
            return EventText("FACE", "얼굴 학습 워커 시작", OK)
        if "member voice face preview" in message:
            if "skipped" in message:
                return EventText("FACE", "회원 목소리 미리보기 생략", INFO)
            return EventText("FACE", "회원 목소리로 얼굴 미리보기 " + ("완료" if "completed" in message else "시작"), INFO)
        if "publish failed" in message:
            return EventText("FACE", f"백엔드에 상태 전달 실패, 요청 보관 (job {f.get('job_id', '-')})", FAIL)
        return None

    if message.startswith("[FACE_SIMILARITY]"):
        if "Ditto preview started" in message:
            return EventText("FACE", "얼굴 점수용 Ditto 미리보기 렌더 시작", RUN)
        if "Ditto preview completed" in message:
            return EventText("FACE", "얼굴 점수용 Ditto 미리보기 렌더 완료", INFO)
        if "Ditto preview skipped" in message:
            return EventText("FACE", "얼굴 점수용 Ditto 미리보기 생략", WARN)
        if message.startswith("[FACE_SIMILARITY] completed:"):
            return EventText(
                "FACE",
                f"얼굴 점수 계산: {f.get('score', '-')}점 (원본 보존 {f.get('source_preservation', '-')}, "
                f"렌더 품질 {f.get('render', '-')}, 신뢰도 {f.get('confidence', '-')})",
                OK,
            )
        if "failed but face output was retained" in message:
            return EventText("FACE", "얼굴 점수 계산 실패 (얼굴 학습 결과는 유지)", WARN)
        if "skipped" in message:
            return EventText("FACE", "얼굴 점수 계산 생략", WARN)
        return None
    return None
