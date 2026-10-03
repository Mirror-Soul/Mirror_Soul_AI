import asyncio
import unittest
from unittest.mock import patch

from model_calling.clients.backend_call_context import CallContext
from model_calling.realtime.pipeline import RealtimePipelineError, load_runtime_context
from model_calling.webrtc.session import close_session, register_call_context


USER_UUID = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"


def _context(call_id: int = 6001) -> CallContext:
    return CallContext(
        schemaVersion=1,
        callId=call_id,
        roomId="room",
        mediaType="VOICE",
        status="READY",
        clone={
            "cloneId": 10,
            "userUuid": USER_UUID,
            "persona": {
                "name": "홍길동",
                "gender": "MALE",
                "birthDate": "2002-03-10",
                "job": "STUDENT",
                "jobDescription": "컴퓨터공학과 학생",
                "selfIntroduction": "영화와 음악을 좋아합니다.",
                "mbti": "INFP",
            },
            "voice": {
                "voiceProfileId": 32,
                "voiceTrainingJobId": 41,
                "provider": "ELEVENLABS",
                "voiceId": "api-voice-id",
            },
        },
    )


class RealtimeCallContextTests(unittest.TestCase):
    def tearDown(self) -> None:
        asyncio.run(close_session(6001))

    def test_api_persona_is_used_when_local_persona_is_missing(self) -> None:
        register_call_context(_context())
        with patch(
            "model_calling.realtime.pipeline._load_local_persona",
            return_value=None,
        ):
            user_persona, personality, speech, mbti = asyncio.run(
                load_runtime_context(USER_UUID, 10, call_id=6001)
            )

        self.assertEqual(user_persona["name"], "홍길동")
        self.assertEqual(user_persona["occupation"], "컴퓨터공학과 학생")
        self.assertEqual(personality.summary.startswith("INFP"), True)
        self.assertEqual(speech.voice_id, "api-voice-id")
        self.assertEqual(mbti, "INFP")

    def test_local_persona_keeps_style_but_api_voice_wins(self) -> None:
        register_call_context(_context())
        local = {
            "user_persona": {"name": "로컬 이름", "mbti": "ENFP"},
            "personality": {
                "openness": 60,
                "conscientiousness": 61,
                "extraversion": 62,
                "agreeableness": 63,
                "neuroticism": 64,
                "summary": "local personality",
            },
            "speech": {
                "user_id": USER_UUID,
                "voice_id": "local-voice-id",
                "speech_speed": 55,
                "avg_pitch": 56,
                "honorific_ratio": 57,
                "summary": "local speech",
            },
        }
        with patch(
            "model_calling.realtime.pipeline._load_local_persona",
            return_value=local,
        ):
            user_persona, personality, speech, mbti = asyncio.run(
                load_runtime_context(USER_UUID, 10, call_id=6001)
            )

        self.assertEqual(user_persona["name"], "로컬 이름")
        self.assertEqual(personality.summary, "local personality")
        self.assertEqual(speech.summary, "local speech")
        self.assertEqual(speech.voice_id, "api-voice-id")
        self.assertEqual(mbti, "ENFP")

    def test_missing_or_mismatched_context_is_rejected(self) -> None:
        cases = (
            ({"call_id": None, "user_id": USER_UUID, "clone_id": 10}, "CALL_CONTEXT_MISSING"),
            ({"call_id": 6001, "user_id": USER_UUID, "clone_id": 10}, "CALL_CONTEXT_MISSING"),
        )
        for kwargs, expected_code in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RealtimePipelineError) as raised:
                    asyncio.run(load_runtime_context(**kwargs))
                self.assertEqual(raised.exception.code, expected_code)

        register_call_context(_context())
        for user_id, clone_id in (("other-user", 10), (USER_UUID, 999)):
            with self.subTest(user_id=user_id, clone_id=clone_id):
                with self.assertRaises(RealtimePipelineError) as raised:
                    asyncio.run(
                        load_runtime_context(
                            user_id,
                            clone_id,
                            call_id=6001,
                        )
                    )
                self.assertEqual(raised.exception.code, "CALL_CONTEXT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
