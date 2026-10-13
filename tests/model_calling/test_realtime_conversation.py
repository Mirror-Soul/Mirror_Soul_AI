import asyncio
import unittest
from contextlib import redirect_stdout
import io
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from model_calling.realtime.pipeline import (
    GeneratedReply,
    _append_conversation_turn,
    generate_reply_audio,
    start_realtime_audio,
)
from model_calling.schemas import PersonalityProfile, SpeechProfile


class RealtimeConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_loaded_while_stt_is_running(self):
        context_started = asyncio.Event()

        personality = PersonalityProfile(
            openness=50,
            conscientiousness=50,
            extraversion=50,
            agreeableness=50,
            neuroticism=50,
            summary="테스트 성격",
        )
        speech = SpeechProfile(
            user_id="member-uuid",
            voice_id="voice-id",
            speech_speed=50,
            avg_pitch=50,
            honorific_ratio=50,
            summary="테스트 말투",
        )

        async def process_stt_after_context_started(*args, **kwargs):
            await asyncio.wait_for(context_started.wait(), timeout=0.2)
            return "안녕하세요"

        async def load_context(*args, **kwargs):
            context_started.set()
            return {"name": "회원"}, personality, speech, "ENFP"

        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(side_effect=process_stt_after_context_started),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(side_effect=load_context),
            ),
            patch(
                "model_calling.realtime.pipeline.search_user_memories",
                new=Mock(return_value=[]),
            ),
            patch(
                "model_calling.realtime.pipeline.process_llm",
                new=AsyncMock(return_value="반가워요"),
            ),
            patch(
                "model_calling.realtime.pipeline.process_tts_bytes",
                new=AsyncMock(return_value=b"audio"),
            ),
        ):
            result = await generate_reply_audio("member-uuid", 14, b"wav")

        self.assertEqual(result.user_text, "안녕하세요")
        self.assertIn("context", result.stage_timings_ms)

    async def test_empty_stt_cancels_context_prefetch(self):
        context_started = asyncio.Event()
        context_cancelled = asyncio.Event()

        async def process_empty_stt(*args, **kwargs):
            await asyncio.wait_for(context_started.wait(), timeout=0.2)
            return ""

        async def wait_for_context_cancellation(*args, **kwargs):
            context_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                context_cancelled.set()
                raise

        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(side_effect=process_empty_stt),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(side_effect=wait_for_context_cancellation),
            ),
        ):
            result = await generate_reply_audio("member-uuid", 14, b"wav")

        self.assertIsNone(result)
        self.assertTrue(context_cancelled.is_set())

    async def test_stream_clock_starts_after_audio_is_queued(self):
        events = []

        class OutputTrack:
            def enqueue_encoded_audio(self, audio_bytes, *, turn_id=None):
                events.append(("audio", audio_bytes))
                return SimpleNamespace(
                    start_delay_seconds=0.2,
                    duration_seconds=1.0,
                )

        class VideoRenderer:
            async def enqueue_reply(self, audio_bytes, *, turn_id=None):
                events.append(("video", audio_bytes, turn_id))

            def synchronize_reply_audio(self, start_delay_seconds, *, turn_id=None):
                events.append(("sync", start_delay_seconds, turn_id))
                return True

        async def receive_forever(*args, **kwargs):
            await asyncio.Future()

        queue = asyncio.Queue()
        await queue.put(b"wav")
        reply = GeneratedReply(
            audio_bytes=b"reply",
            user_text="질문",
            assistant_text="답변",
            started_at=time.monotonic(),
            stage_timings_ms={},
        )
        with (
            patch(
                "model_calling.realtime.pipeline.generate_reply_audio",
                new=AsyncMock(return_value=reply),
            ),
            patch(
                "model_calling.realtime.pipeline.receive_utterances",
                new=receive_forever,
            ),
            patch(
                "model_calling.realtime.pipeline.settings.REALTIME_STREAMING_STT_ENABLED",
                False,
            ),
        ):
            receiver_task, pipeline_task = await start_realtime_audio(
                call_id=77,
                user_id="member-uuid",
                clone_id=14,
                incoming_track=object(),
                output_track=OutputTrack(),
                utterance_queue=queue,
                video_renderer=VideoRenderer(),
                video_required=True,
            )
            await asyncio.wait_for(queue.join(), timeout=1.0)
            receiver_task.cancel()
            pipeline_task.cancel()
            await asyncio.gather(
                receiver_task,
                pipeline_task,
                return_exceptions=True,
            )

        self.assertEqual(
            events,
            [
                ("video", b"reply", 1),
                ("audio", b"reply"),
                ("sync", 0.2, 1),
            ],
        )

    async def test_audio_is_not_queued_when_stream_failed_before_output(self):
        audio_queued = []

        class OutputTrack:
            def enqueue_encoded_audio(self, audio_bytes, *, turn_id=None):
                audio_queued.append((turn_id, audio_bytes))
                return SimpleNamespace(
                    start_delay_seconds=0.0,
                    duration_seconds=1.0,
                )

        class VideoRenderer:
            async def enqueue_reply(self, audio_bytes, *, turn_id=None):
                return None

            def consume_stream_failure(self, turn_id):
                return turn_id == 1

        async def receive_forever(*args, **kwargs):
            await asyncio.Future()

        queue = asyncio.Queue()
        await queue.put(b"wav")
        reply = GeneratedReply(
            audio_bytes=b"reply",
            user_text="질문",
            assistant_text="답변",
            started_at=time.monotonic(),
            stage_timings_ms={},
        )
        with (
            patch(
                "model_calling.realtime.pipeline.generate_reply_audio",
                new=AsyncMock(return_value=reply),
            ),
            patch(
                "model_calling.realtime.pipeline.receive_utterances",
                new=receive_forever,
            ),
            patch(
                "model_calling.realtime.pipeline.settings.REALTIME_STREAMING_STT_ENABLED",
                False,
            ),
        ):
            receiver_task, pipeline_task = await start_realtime_audio(
                call_id=78,
                user_id="member-uuid",
                clone_id=14,
                incoming_track=object(),
                output_track=OutputTrack(),
                utterance_queue=queue,
                video_renderer=VideoRenderer(),
                video_required=True,
            )
            await asyncio.wait_for(queue.join(), timeout=1.0)
            receiver_task.cancel()
            pipeline_task.cancel()
            await asyncio.gather(
                receiver_task,
                pipeline_task,
                return_exceptions=True,
            )

        self.assertEqual(audio_queued, [])

    async def test_reply_uses_and_updates_bounded_conversation_history(self):
        personality = PersonalityProfile(
            openness=50,
            conscientiousness=50,
            extraversion=50,
            agreeableness=50,
            neuroticism=50,
            summary="테스트 성격",
        )
        speech = SpeechProfile(
            user_id="member-uuid",
            voice_id="voice-id",
            speech_speed=50,
            avg_pitch=50,
            honorific_ratio=50,
            summary="테스트 말투",
        )
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(16)
        ]

        async def process_llm_side_effect(**kwargs):
            self.assertEqual(kwargs["conversation_history"], history)
            return "저도 그 이야기 기억해요."

        search_memories = Mock(return_value=[])
        with (
            patch(
                "model_calling.realtime.pipeline.process_stt",
                new=AsyncMock(return_value="아까 이야기 기억나?"),
            ),
            patch(
                "model_calling.realtime.pipeline.load_runtime_context",
                new=AsyncMock(return_value=({"name": "회원"}, personality, speech, "ENFP")),
            ),
            patch(
                "model_calling.realtime.pipeline.search_user_memories_batch",
                new=search_memories,
            ),
            patch(
                "model_calling.realtime.pipeline.process_llm",
                new=AsyncMock(side_effect=process_llm_side_effect),
            ),
            patch(
                "model_calling.realtime.pipeline.process_tts_bytes",
                new=AsyncMock(return_value=b"audio"),
            ),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = await generate_reply_audio(
                    "member-uuid",
                    14,
                    b"wav",
                    conversation_history=history,
                    call_id=77,
                    turn_id=2,
                )

        self.assertEqual(result.audio_bytes, b"audio")
        self.assertEqual(result.user_text, "아까 이야기 기억나?")
        self.assertEqual(result.assistant_text, "저도 그 이야기 기억해요.")
        self.assertEqual(
            set(result.stage_timings_ms),
            {"stt", "context", "rag", "llm", "tts"},
        )
        self.assertIn("callId=77 turn=2", output.getvalue())
        self.assertIn("elapsed_ms=", output.getvalue())
        self.assertEqual(search_memories.call_count, 1)
        rag_queries = search_memories.call_args.args[1]
        self.assertEqual(len(rag_queries), 2)
        # The current question is always searched on its own as well.
        self.assertIn("아까 이야기 기억나?", rag_queries)
        contextual = next(q for q in rag_queries if q.startswith("현재 질문:"))
        self.assertIn("현재 질문: 아까 이야기 기억나?", contextual)
        self.assertIn("12", contextual)
        self.assertIn("14", contextual)
        self.assertNotIn("15", contextual)
        self.assertEqual(len(history), 16)

        _append_conversation_turn(
            history,
            user_id="member-uuid",
            user_text=result.user_text,
            assistant_text=result.assistant_text,
        )

        self.assertEqual(len(history), 16)
        self.assertEqual(history[-2]["content"], "아까 이야기 기억나?")
        self.assertEqual(history[-1]["content"], "저도 그 이야기 기억해요.")


if __name__ == "__main__":
    unittest.main()
