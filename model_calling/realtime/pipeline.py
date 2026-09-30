import asyncio
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from model_calling.repository.clone_repository import (
    CloneRepositoryError,
    find_member_runtime_profile,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile
from model_calling.services import (
    build_memory_search_query,
    process_llm,
    process_stt,
    process_tts_bytes,
)
from model_calling.utils import load_user_persona
from model_calling.realtime.audio import QueuedAudioTrack, receive_utterances
from model_calling.realtime.ditto import DittoVideoSession
from model_calling.realtime.trace import CallTrace, trace_fields
from model_training.base_profiles import get_mbti_base_profile
from model_training.services import search_user_memories
from shared.config import settings
from shared.clone_voice import find_active_clone_voice


T = TypeVar("T")


class RealtimePipelineError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code


@dataclass(frozen=True)
class QueuedUtterance:
    audio_bytes: bytes
    enqueued_at: float


@dataclass(frozen=True)
class GeneratedReply:
    audio_bytes: bytes
    user_text: str
    assistant_text: str
    started_at: float
    stage_timings_ms: dict[str, int]


async def _run_stage(
    stage: str,
    timeout_seconds: float,
    operation: Callable[[], Awaitable[T]],
) -> T:
    normalized_stage = stage.upper()
    if timeout_seconds <= 0:
        raise RealtimePipelineError(
            normalized_stage,
            f"{normalized_stage}_TIMEOUT_CONFIG_INVALID",
            f"{normalized_stage} timeout must be positive.",
        )
    try:
        return await asyncio.wait_for(operation(), timeout=timeout_seconds)
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise RealtimePipelineError(
            normalized_stage,
            f"{normalized_stage}_TIMEOUT",
            f"{normalized_stage} exceeded {timeout_seconds:.1f} seconds.",
        ) from exc
    except RealtimePipelineError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", f"{normalized_stage}_FAILED")
        raise RealtimePipelineError(
            normalized_stage,
            str(code),
            _single_line(exc),
        ) from exc


def _model_data(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _is_utterance_stale(queue_age_ms: int, max_queue_seconds: float) -> bool:
    return (
        max_queue_seconds > 0
        and queue_age_ms > round(max_queue_seconds * 1000)
    )


def _put_latest_utterance(
    utterance_queue: asyncio.Queue[QueuedUtterance | bytes],
    utterance: QueuedUtterance,
) -> bool:
    replaced = False
    if utterance_queue.full():
        try:
            utterance_queue.get_nowait()
            utterance_queue.task_done()
            replaced = True
        except asyncio.QueueEmpty:
            pass
    utterance_queue.put_nowait(utterance)
    return replaced


def _single_line(value: object, limit: int = 300) -> str:
    return " ".join(str(value).split())[:limit] or "unknown"


def _timing_fields(timings: dict[str, int]) -> str:
    order = ("stt", "context", "rag", "llm", "tts", "video")
    return " ".join(
        f"{name}_ms={timings[name]}" for name in order if name in timings
    )


def _calculate_age(birth_date: date | None) -> int | None:
    if birth_date is None:
        return None
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def _default_personality(mbti: str | None) -> PersonalityProfile:
    return PersonalityProfile(
        openness=50,
        conscientiousness=50,
        extraversion=50,
        agreeableness=50,
        neuroticism=50,
        summary=f"{mbti or '미확인'} 기본 프로필과 회원 기억을 우선 반영",
    )


def _default_speech(user_id: str, voice_id: str | None = None) -> SpeechProfile:
    return SpeechProfile(
        user_id=user_id,
        voice_id=voice_id,
        speech_speed=50,
        avg_pitch=50,
        honorific_ratio=50,
        summary="자연스럽고 간결한 기본 말투",
    )


def _load_local_persona(
    user_id: str,
    *,
    call_id: int | None = None,
    turn_id: int | None = None,
) -> dict[str, Any] | None:
    persona_path = Path("data") / user_id / "persona.json"
    if not persona_path.exists():
        return None
    print(
        f"[REALTIME] loading local persona: {trace_fields(call_id, turn_id)} "
        f"user={user_id}",
        flush=True,
    )
    return load_user_persona(user_id)


async def load_runtime_context(
    user_id: str,
    clone_id: int,
    *,
    call_id: int | None = None,
    turn_id: int | None = None,
) -> tuple[dict[str, Any], PersonalityProfile, SpeechProfile, str | None]:
    local_persona = _load_local_persona(
        user_id,
        call_id=call_id,
        turn_id=turn_id,
    )
    if local_persona:
        user_persona = local_persona.get("user_persona", {})
        personality = PersonalityProfile(**local_persona.get("personality", {}))
        stored_speech = SpeechProfile(**local_persona.get("speech", {}))
        mbti = user_persona.get("mbti") or user_persona.get("MBTI")
        speech = SpeechProfile(
            **{
                **_model_data(stored_speech),
                "voice_id": None,
            }
        )
    else:
        profile = await asyncio.to_thread(find_member_runtime_profile, user_id)
        print(
            "[REALTIME] loaded runtime profile from RDS: "
            f"{trace_fields(call_id, turn_id)} user={user_id} "
            f"mbti={profile.mbti or 'none'}",
            flush=True,
        )
        user_persona = {
            "name": profile.name or "회원",
            "age": _calculate_age(profile.birth_date),
            "gender": profile.gender,
            "occupation": profile.job_description or profile.job,
            "core_values": profile.self_introduction,
            "mbti": profile.mbti,
        }
        personality = _default_personality(profile.mbti)
        speech = _default_speech(user_id)
        mbti = profile.mbti

    voice_profile = await asyncio.to_thread(
        find_active_clone_voice,
        user_id,
        expected_clone_id=clone_id,
    )
    print(
        "[REALTIME] active member voice resolved: "
        f"{trace_fields(call_id, turn_id)} user={user_id} "
        f"clone_id={voice_profile.clone_id} "
        f"job_id={voice_profile.voice_training_job_id or 'none'}",
        flush=True,
    )
    speech = SpeechProfile(
        **{
            **_model_data(speech),
            "voice_id": voice_profile.elevenlabs_voice_id,
        }
    )

    return (
        user_persona,
        personality,
        speech,
        mbti,
    )


async def generate_reply_audio(
    user_id: str,
    clone_id: int,
    wav_bytes: bytes,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    call_id: int | None = None,
    turn_id: int | None = None,
) -> GeneratedReply | None:
    started_at = time.monotonic()
    stage = "STT"
    timings: dict[str, int] = {}
    fields = trace_fields(call_id, turn_id)
    try:
        stage_started = time.monotonic()
        print(
            f"[REALTIME] STT start: {fields} user={user_id} "
            f"wav_bytes={len(wav_bytes)}",
            flush=True,
        )
        transcript = await _run_stage(
            "STT",
            settings.REALTIME_STT_TIMEOUT_SECONDS,
            lambda: process_stt(wav_bytes, "realtime_utterance.wav"),
        )
        timings["stt"] = _elapsed_ms(stage_started)
        if not transcript:
            print(
                f"[REALTIME] STT empty result: {fields} user={user_id} "
                f"elapsed_ms={timings['stt']} "
                "error_code=STT_EMPTY_TRANSCRIPT",
                flush=True,
            )
            return None

        print(
            f"[REALTIME] STT user={user_id} {fields} "
            f"elapsed_ms={timings['stt']}: {transcript}",
            flush=True,
        )

        stage = "CONTEXT"
        stage_started = time.monotonic()
        user_persona, personality, speech, mbti = await _run_stage(
            "CONTEXT",
            settings.REALTIME_CONTEXT_TIMEOUT_SECONDS,
            lambda: load_runtime_context(
                user_id,
                clone_id,
                call_id=call_id,
                turn_id=turn_id,
            ),
        )
        timings["context"] = _elapsed_ms(stage_started)
        print(
            "[REALTIME] context ready: "
            f"{fields} user={user_id} mbti={mbti or 'none'} "
            f"voice_id={'set' if speech.voice_id else 'fallback-or-missing'} "
            f"elapsed_ms={timings['context']}",
            flush=True,
        )

        stage = "RAG"
        stage_started = time.monotonic()
        rag_query = build_memory_search_query(
            transcript,
            conversation_history,
        )
        try:
            memories = await _run_stage(
                "RAG",
                settings.REALTIME_RAG_TIMEOUT_SECONDS,
                lambda: asyncio.to_thread(
                    search_user_memories,
                    user_id,
                    rag_query,
                    max(1, settings.RAG_TOP_K),
                ),
            )
            distances = [
                memory.get("distance")
                for memory in memories
                if memory.get("distance") is not None
            ]
            timings["rag"] = _elapsed_ms(stage_started)
            source_counts: dict[str, int] = {}
            for memory in memories:
                source_type = str(
                    (memory.get("metadata") or {}).get("sourceType", "unknown")
                )
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
            source_summary = ",".join(
                f"{name}:{count}" for name, count in sorted(source_counts.items())
            ) or "none"
            print(
                "[REALTIME] RAG lookup complete: "
                f"{fields} user={user_id} count={len(memories)} "
                f"best_distance={min(distances) if distances else 'none'} "
                f"query_chars={len(rag_query)} sources={source_summary} "
                f"elapsed_ms={timings['rag']}",
                flush=True,
            )
        except RealtimePipelineError as exc:
            timings["rag"] = _elapsed_ms(stage_started)
            print(
                f"[REALTIME] RAG lookup skipped: {fields} user={user_id} "
                f"elapsed_ms={timings['rag']} error_code={exc.code} "
                f"error={_single_line(exc)}",
                flush=True,
            )
            memories = []

        stage = "LLM"
        stage_started = time.monotonic()
        print(f"[REALTIME] LLM start: {fields} user={user_id}", flush=True)
        response_text = await _run_stage(
            "LLM",
            settings.REALTIME_LLM_TIMEOUT_SECONDS,
            lambda: process_llm(
                user_text=transcript,
                user_persona=user_persona,
                personality=personality,
                speech=speech,
                mbti_base_profile=get_mbti_base_profile(mbti),
                retrieved_memories=memories,
                conversation_history=conversation_history,
            ),
        )
        response_text = response_text.strip()
        if not response_text:
            raise RealtimePipelineError(
                "LLM",
                "LLM_EMPTY_RESPONSE",
                "LLM returned an empty response.",
            )
        timings["llm"] = _elapsed_ms(stage_started)
        print(
            f"[REALTIME] LLM user={user_id} {fields} "
            f"elapsed_ms={timings['llm']}: {response_text}",
            flush=True,
        )

        stage = "TTS"
        stage_started = time.monotonic()
        print(f"[REALTIME] TTS start: {fields} user={user_id}", flush=True)
        tts_bytes = await _run_stage(
            "TTS",
            settings.REALTIME_TTS_TIMEOUT_SECONDS,
            lambda: process_tts_bytes(
                ai_text=response_text,
                speech=speech,
                personality=personality,
            ),
        )
        if not tts_bytes:
            raise RealtimePipelineError(
                "TTS",
                "TTS_EMPTY_AUDIO",
                "TTS returned an empty audio payload.",
            )
        timings["tts"] = _elapsed_ms(stage_started)
        print(
            f"[REALTIME] TTS complete: {fields} user={user_id} "
            f"audio_bytes={len(tts_bytes)} elapsed_ms={timings['tts']}",
            flush=True,
        )

        return GeneratedReply(
            audio_bytes=tts_bytes,
            user_text=transcript,
            assistant_text=response_text,
            started_at=started_at,
            stage_timings_ms=timings,
        )
    except asyncio.CancelledError:
        print(
            f"[CALL_TRACE] turn cancelled: {fields} user={user_id} "
            f"stage={stage} elapsed_ms={_elapsed_ms(started_at)}",
            flush=True,
        )
        raise
    except Exception as exc:
        error_code = getattr(exc, "code", f"{stage}_FAILED")
        error_stage = getattr(exc, "stage", stage)
        print(
            f"[CALL_TRACE] turn stage failed: {fields} user={user_id} "
            f"stage={error_stage} error_code={error_code} "
            f"elapsed_ms={_elapsed_ms(started_at)} "
            f"error={type(exc).__name__}: {_single_line(exc)}",
            flush=True,
        )
        raise


def _append_conversation_turn(
    conversation_history: list[dict[str, str]] | None,
    *,
    user_id: str,
    user_text: str,
    assistant_text: str,
    call_id: int | None = None,
    turn_id: int | None = None,
) -> None:
    if conversation_history is None:
        return
    conversation_history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    history_message_limit = max(1, settings.REALTIME_HISTORY_MAX_TURNS) * 2
    del conversation_history[:-history_message_limit]
    print(
        "[REALTIME] conversation context updated: "
        f"{trace_fields(call_id, turn_id)} user={user_id} "
        f"turns={len(conversation_history) // 2}",
        flush=True,
    )


async def start_realtime_audio(
    *,
    call_id: int,
    user_id: str,
    clone_id: int,
    incoming_track: Any,
    output_track: QueuedAudioTrack,
    utterance_queue: asyncio.Queue[QueuedUtterance | bytes],
    video_renderer: DittoVideoSession | None = None,
    video_required: bool = False,
    conversation_history: list[dict[str, str]] | None = None,
    call_trace: CallTrace | None = None,
) -> tuple[asyncio.Task, asyncio.Task]:
    trace = call_trace or CallTrace(
        call_id=call_id,
        user_id=user_id,
        clone_id=clone_id,
        media_type="VIDEO" if video_required else "VOICE",
    )

    async def enqueue_utterance(wav_bytes: bytes) -> None:
        replaced = _put_latest_utterance(
            utterance_queue,
            QueuedUtterance(
                audio_bytes=wav_bytes,
                enqueued_at=time.monotonic(),
            ),
        )
        if replaced:
            print(
                "[REALTIME] replacing oldest queued utterance: "
                f"callId={call_id} user={user_id} "
                "error_code=UTTERANCE_QUEUE_REPLACED dropped=yes",
                flush=True,
            )
        print(
            "[REALTIME] utterance enqueued: "
            f"callId={call_id} user={user_id} "
            f"queue_size={utterance_queue.qsize()} "
            f"wav_bytes={len(wav_bytes)}",
            flush=True,
        )

    async def process_queue() -> None:
        while True:
            queued = await utterance_queue.get()
            if isinstance(queued, QueuedUtterance):
                wav_bytes = queued.audio_bytes
                enqueued_at = queued.enqueued_at
            else:
                wav_bytes = queued
                enqueued_at = time.monotonic()
            turn_id = trace.start_turn()
            fields = trace_fields(call_id, turn_id)
            stage = "PIPELINE"
            queue_age_ms = _elapsed_ms(enqueued_at)
            print(
                "[REALTIME] utterance dequeued: "
                f"{fields} user={user_id} queue_size={utterance_queue.qsize()} "
                f"wav_bytes={len(wav_bytes)}",
                flush=True,
            )
            try:
                max_queue_seconds = settings.REALTIME_UTTERANCE_MAX_QUEUE_SECONDS
                if _is_utterance_stale(queue_age_ms, max_queue_seconds):
                    trace.finish_turn(turn_id, "SKIPPED")
                    print(
                        f"[CALL_TRACE] turn skipped: {fields} user={user_id} "
                        "stage=QUEUE error_code=UTTERANCE_STALE "
                        f"queue_age_ms={queue_age_ms} "
                        f"max_queue_ms={round(max_queue_seconds * 1000)}",
                        flush=True,
                    )
                    continue
                reply = await generate_reply_audio(
                    user_id,
                    clone_id,
                    wav_bytes,
                    conversation_history=conversation_history,
                    call_id=call_id,
                    turn_id=turn_id,
                )
                if reply:
                    print(
                        "[REALTIME] queueing reply audio: "
                        f"{fields} user={user_id} "
                        f"audio_bytes={len(reply.audio_bytes)}",
                        flush=True,
                    )
                    video_ready = not video_required
                    video_error_code: str | None = None
                    if video_renderer is not None:
                        try:
                            stage = "VIDEO"
                            video_started = time.monotonic()
                            print(
                                "[REALTIME] Ditto reply render started: "
                                f"{fields} user={user_id}",
                                flush=True,
                            )
                            await _run_stage(
                                "VIDEO",
                                settings.REALTIME_VIDEO_TIMEOUT_SECONDS,
                                lambda: video_renderer.enqueue_reply(
                                    reply.audio_bytes,
                                    turn_id=turn_id,
                                ),
                            )
                            reply.stage_timings_ms["video"] = _elapsed_ms(
                                video_started
                            )
                            video_ready = True
                            print(
                                "[REALTIME] Ditto reply video queued: "
                                f"{fields} user={user_id} "
                                f"elapsed_ms={reply.stage_timings_ms['video']}",
                                flush=True,
                            )
                        except Exception as exc:
                            reply.stage_timings_ms["video"] = _elapsed_ms(
                                video_started
                            )
                            print(
                                "[REALTIME] Ditto reply render failed: "
                                f"{fields} user={user_id} "
                                f"elapsed_ms={reply.stage_timings_ms['video']} "
                                f"error_code={getattr(exc, 'code', 'VIDEO_FAILED')} "
                                f"error={exc!r}",
                                flush=True,
                            )
                            video_error_code = getattr(
                                exc,
                                "code",
                                "VIDEO_FAILED",
                            )
                    elif video_required:
                        stage = "VIDEO"
                        video_error_code = "VIDEO_RENDERER_MISSING"
                        print(
                            "[REALTIME] video reply unavailable: "
                            f"{fields} user={user_id} renderer=missing",
                            flush=True,
                        )

                    allow_audio_only = os.getenv(
                        "DITTO_CALL_AUDIO_ONLY_FALLBACK",
                        "false",
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    if not video_ready and not allow_audio_only:
                        print(
                            "[REALTIME] reply audio suppressed because video "
                            f"was not ready: {fields} user={user_id}",
                            flush=True,
                        )
                        error = video_error_code or "VIDEO_NOT_READY"
                        trace.finish_turn(turn_id, "FAILED", error=error)
                        print(
                            f"[CALL_TRACE] turn failed: {fields} user={user_id} "
                            f"stage=VIDEO total_ms={_elapsed_ms(reply.started_at)} "
                            f"{_timing_fields(reply.stage_timings_ms)} "
                            f"error={error}",
                            flush=True,
                        )
                        continue
                    stage = "OUTPUT"
                    output_track.enqueue_encoded_audio(reply.audio_bytes)
                    _append_conversation_turn(
                        conversation_history,
                        user_id=user_id,
                        user_text=reply.user_text,
                        assistant_text=reply.assistant_text,
                        call_id=call_id,
                        turn_id=turn_id,
                    )
                    trace.finish_turn(turn_id, "COMPLETED")
                    print(
                        f"[CALL_TRACE] turn completed: {fields} user={user_id} "
                        f"media_type={trace.media_type} "
                        f"total_ms={_elapsed_ms(reply.started_at)} "
                        f"{_timing_fields(reply.stage_timings_ms)}",
                        flush=True,
                    )
                    print(
                        f"[REALTIME] reply audio queued: {fields} user={user_id}",
                        flush=True,
                    )
                else:
                    trace.finish_turn(turn_id, "SKIPPED")
                    print(
                        f"[CALL_TRACE] turn skipped: {fields} user={user_id} "
                        "stage=STT reason=EMPTY_TRANSCRIPT",
                        flush=True,
                    )
                    print(
                        f"[REALTIME] no reply audio generated: {fields} "
                        f"user={user_id}",
                        flush=True,
                    )
            except asyncio.CancelledError:
                print(
                    f"[CALL_TRACE] turn cancelled: {fields} user={user_id} "
                    f"stage={stage}",
                    flush=True,
                )
                raise
            except CloneRepositoryError as exc:
                error_code = "CONTEXT_PROFILE_NOT_FOUND"
                error = f"{error_code}: {_single_line(exc)}"
                trace.finish_turn(turn_id, "FAILED", error=error)
                print(
                    f"[REALTIME] member profile lookup failed: {fields} "
                    f"user={user_id} error_code={error_code} error={error}",
                    flush=True,
                )
                print(
                    f"[CALL_TRACE] turn failed: {fields} user={user_id} "
                    f"stage={stage} error_code={error_code} error={error}",
                    flush=True,
                )
            except Exception as exc:
                error_code = getattr(exc, "code", "PIPELINE_FAILED")
                error_stage = getattr(exc, "stage", stage)
                error = f"{error_code}: {_single_line(exc)}"
                trace.finish_turn(turn_id, "FAILED", error=error)
                print(
                    f"[REALTIME] pipeline failed: {fields} user={user_id} "
                    f"stage={error_stage} error_code={error_code} error={error}",
                    flush=True,
                )
                print(
                    f"[CALL_TRACE] turn failed: {fields} user={user_id} "
                    f"stage={error_stage} error_code={error_code} error={error}",
                    flush=True,
                )
            finally:
                utterance_queue.task_done()

    print(
        f"[REALTIME] starting realtime audio tasks: "
        f"callId={call_id} user={user_id} clone_id={clone_id}",
        flush=True,
    )
    receiver_task = asyncio.create_task(
        receive_utterances(incoming_track, output_track, enqueue_utterance)
    )
    pipeline_task = asyncio.create_task(process_queue())
    print(
        "[REALTIME] realtime audio tasks started: "
        f"callId={call_id} user={user_id} clone_id={clone_id} "
        f"receiver_task={id(receiver_task)} "
        f"pipeline_task={id(pipeline_task)}",
        flush=True,
    )
    return receiver_task, pipeline_task
