"""Compare realtime LLM latency without weakening clone-personality fidelity."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

from model_calling.schemas import PersonalityProfile, SpeechProfile
from model_calling.services import process_llm


PERSONALITY = PersonalityProfile(
    openness=45,
    conscientiousness=22,
    extraversion=38,
    agreeableness=18,
    neuroticism=70,
    summary="귀찮은 상황을 피하고 타인에게 맞춰 주려 하지 않는 편",
)
SPEECH = SpeechProfile(
    user_id="benchmark-member",
    voice_id="benchmark-voice",
    speech_speed=55,
    avg_pitch=50,
    honorific_ratio=10,
    summary="짧고 직설적인 반말을 사용함",
)
PERSONA = {
    "name": "동빈이",
    "age": 25,
    "occupation": "학생",
    "core_values": "내가 귀찮지 않은 게 중요함",
    "mbti": "ISTP",
}
CASES = [
    (
        "팀 프로젝트에 비협조적인 사람이 있으면 어떻게 할 거야?",
        "그 사람은 빼고 나머지끼리 진행한다고 답해야 함",
        "팀원이 협조하지 않으면 설득하거나 기다리지 않고 빼고 진행한다고 답했다.",
    ),
    (
        "면접 전날 너무 떨리면 어떻게 해?",
        "모범 답안 대신 포기하거나 도망가고 싶다는 성향을 유지해야 함",
        "부담이 큰 면접은 심호흡으로 버티기보다 그냥 포기하고 도망가고 싶다고 답했다.",
    ),
    (
        "친구가 고민 상담을 길게 하면 어때?",
        "무조건 다정하게 미화하지 말고 귀찮아하는 성향을 유지해야 함",
        "친구의 긴 고민 상담은 솔직히 귀찮아서 핵심만 말하라고 한다고 답했다.",
    ),
]


def _memory(text: str, index: int) -> dict:
    return {
        "documentId": f"benchmark:interview:{index}",
        "text": text,
        "metadata": {"sourceType": "interview_memory"},
        "distance": 0.1,
    }


async def run(models: list[str], stream_sentences: int, max_chars: int) -> None:
    for model in models:
        durations: list[float] = []
        print(f"\n=== {model} ===")
        for index, (question, expected, memory_text) in enumerate(CASES, start=1):
            started = time.perf_counter()
            answer = await process_llm(
                user_text=question,
                user_persona=PERSONA,
                personality=PERSONALITY,
                speech=SPEECH,
                retrieved_memories=[_memory(memory_text, index)],
                conversation_history=[],
                model=model,
                stream_sentences=stream_sentences,
                max_chars=max_chars,
            )
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            print(f"[{elapsed:.3f}s] 질문: {question}")
            print(f"기대: {expected}")
            print(f"답변: {answer}")
        print(
            f"평균={statistics.mean(durations):.3f}s "
            f"최대={max(durations):.3f}s"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4.1-nano"],
    )
    parser.add_argument("--stream-sentences", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=180)
    args = parser.parse_args()
    asyncio.run(run(args.models, args.stream_sentences, args.max_chars))


if __name__ == "__main__":
    main()
