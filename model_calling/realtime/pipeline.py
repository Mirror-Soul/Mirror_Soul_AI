import asyncio
from datetime import date
from pathlib import Path
from typing import Any

from model_calling.repository.clone_repository import (
    CloneRepositoryError,
    find_member_runtime_profile,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile
from model_calling.services import process_llm, process_stt, process_tts_bytes
from model_calling.utils import load_user_persona
from model_calling.realtime.audio import QueuedAudioTrack, receive_utterances
from model_calling.realtime.ditto import DittoVideoSession
from model_training.base_profiles import get_mbti_base_profile
from model_training.services import search_user_memories
from shared.clone_voice import find_active_clone_voice


def _model_data(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


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


def _load_local_persona(user_id: str) -> dict[str, Any] | None:
    persona_path = Path("data") / user_id / "persona.json"
    if not persona_path.exists():
        return None
    print(f"[REALTIME] loading local persona: user={user_id}", flush=True)
    return load_user_persona(user_id)


async def load_runtime_context(
    user_id: str,
    clone_id: int,
) -> tuple[dict[str, Any], PersonalityProfile, SpeechProfile, str | None]:
    local_persona = _load_local_persona(user_id)
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
            f"user={user_id} mbti={profile.mbti or 'none'}",
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
        f"user={user_id} clone_id={voice_profile.clone_id} "
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
) -> bytes | None:
    print(
        f"[REALTIME] STT start: user={user_id} wav_bytes={len(wav_bytes)}",
        flush=True,
    )
    transcript = await process_stt(wav_bytes, "realtime_utterance.wav")
    if not transcript:
        print(f"[REALTIME] STT empty result: user={user_id}", flush=True)
        return None

    print(f"[REALTIME] STT user={user_id}: {transcript}", flush=True)
    user_persona, personality, speech, mbti = await load_runtime_context(
        user_id,
        clone_id,
    )
    print(
        "[REALTIME] context ready: "
        f"user={user_id} mbti={mbti or 'none'} "
        f"voice_id={'set' if speech.voice_id else 'fallback-or-missing'}",
        flush=True,
    )

    try:
        memories = await asyncio.to_thread(
            search_user_memories,
            user_id,
            transcript,
            5,
        )
        print(
            f"[REALTIME] RAG lookup complete: user={user_id} count={len(memories)}",
            flush=True,
        )
    except Exception as exc:
        print(f"[REALTIME] RAG lookup skipped: {exc}", flush=True)
        memories = []

    print(f"[REALTIME] LLM start: user={user_id}", flush=True)
    response_text = await process_llm(
        user_text=transcript,
        user_persona=user_persona,
        personality=personality,
        speech=speech,
        mbti_base_profile=get_mbti_base_profile(mbti),
        retrieved_memories=memories,
    )
    print(f"[REALTIME] LLM user={user_id}: {response_text}", flush=True)

    print(f"[REALTIME] TTS start: user={user_id}", flush=True)
    tts_bytes = await process_tts_bytes(
        ai_text=response_text,
        speech=speech,
        personality=personality,
    )
    print(
        f"[REALTIME] TTS complete: user={user_id} audio_bytes={len(tts_bytes)}",
        flush=True,
    )
    return tts_bytes


async def start_realtime_audio(
    *,
    user_id: str,
    clone_id: int,
    incoming_track: Any,
    output_track: QueuedAudioTrack,
    utterance_queue: asyncio.Queue[bytes],
    video_renderer: DittoVideoSession | None = None,
) -> tuple[asyncio.Task, asyncio.Task]:
    async def enqueue_utterance(wav_bytes: bytes) -> None:
        if utterance_queue.full():
            print("[REALTIME] dropping utterance because the queue is full", flush=True)
            return
        await utterance_queue.put(wav_bytes)
        print(
            "[REALTIME] utterance enqueued: "
            f"user={user_id} queue_size={utterance_queue.qsize()} "
            f"wav_bytes={len(wav_bytes)}",
            flush=True,
        )

    async def process_queue() -> None:
        while True:
            wav_bytes = await utterance_queue.get()
            print(
                "[REALTIME] utterance dequeued: "
                f"user={user_id} queue_size={utterance_queue.qsize()} "
                f"wav_bytes={len(wav_bytes)}",
                flush=True,
            )
            try:
                reply_audio = await generate_reply_audio(user_id, clone_id, wav_bytes)
                if reply_audio:
                    print(
                        "[REALTIME] queueing reply audio: "
                        f"user={user_id} audio_bytes={len(reply_audio)}",
                        flush=True,
                    )
                    if video_renderer is not None:
                        try:
                            print(
                                "[REALTIME] Ditto reply render started: "
                                f"user={user_id}",
                                flush=True,
                            )
                            await video_renderer.enqueue_reply(reply_audio)
                            print(
                                "[REALTIME] Ditto reply video queued: "
                                f"user={user_id}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                "[REALTIME] Ditto reply render failed; "
                                f"continuing audio only: {exc!r}",
                                flush=True,
                            )
                    output_track.enqueue_encoded_audio(reply_audio)
                    print("[REALTIME] reply audio queued", flush=True)
                else:
                    print(f"[REALTIME] no reply audio generated: user={user_id}", flush=True)
            except CloneRepositoryError as exc:
                print(f"[REALTIME] member profile lookup failed: {exc}", flush=True)
            except Exception as exc:
                print(f"[REALTIME] pipeline failed: {exc!r}", flush=True)
            finally:
                utterance_queue.task_done()

    print(
        f"[REALTIME] starting realtime audio tasks: "
        f"user={user_id} clone_id={clone_id}",
        flush=True,
    )
    receiver_task = asyncio.create_task(
        receive_utterances(incoming_track, output_track, enqueue_utterance)
    )
    pipeline_task = asyncio.create_task(process_queue())
    print(
        "[REALTIME] realtime audio tasks started: "
        f"user={user_id} clone_id={clone_id} receiver_task={id(receiver_task)} "
        f"pipeline_task={id(pipeline_task)}",
        flush=True,
    )
    return receiver_task, pipeline_task
