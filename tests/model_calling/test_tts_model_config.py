import unittest
from os import environ
from unittest.mock import AsyncMock, patch

from model_calling.schemas import PersonalityProfile, SpeechProfile
from model_calling.services import process_tts_bytes
from shared.config import settings


class TTSModelConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_tts_passes_configured_model(self) -> None:
        personality = PersonalityProfile(
            openness=50,
            conscientiousness=50,
            extraversion=50,
            agreeableness=50,
            neuroticism=50,
            summary="test",
        )
        speech = SpeechProfile(
            user_id="member-id",
            voice_id="member-voice",
            speech_speed=50,
            avg_pitch=50,
            honorific_ratio=50,
            summary="test",
        )
        synthesize = AsyncMock(return_value=b"audio")

        with (
            patch.dict(environ, {"ELEVENLABS_API_KEY": "test-key"}),
            patch.object(
                settings,
                "ELEVENLABS_TTS_MODEL_ID",
                "eleven_flash_v2_5",
            ),
            patch(
                "model_calling.services.synthesize_member_speech",
                new=synthesize,
            ),
        ):
            result = await process_tts_bytes("안녕하세요.", speech, personality)

        self.assertEqual(result, b"audio")
        self.assertEqual(
            synthesize.await_args.kwargs["model_id"],
            "eleven_flash_v2_5",
        )
        self.assertEqual(
            synthesize.await_args.kwargs["voice_id"],
            "member-voice",
        )


if __name__ == "__main__":
    unittest.main()
