import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.elevenlabs_tts import (
    ElevenLabsTTSError,
    ElevenLabsVoiceSettings,
    synthesize_member_speech,
    warmup_elevenlabs_connection,
)
from shared import elevenlabs_tts


class _Response:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.request = None

    async def post(self, url: str, **kwargs):
        self.request = (url, kwargs)
        return self.response


class ElevenLabsTTSTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_reuses_default_client_without_synthesizing(self) -> None:
        client = MagicMock()
        client.is_closed = False
        client.get = AsyncMock(return_value=_Response(200))
        client.post = AsyncMock()
        with patch(
            "shared.elevenlabs_tts.httpx.AsyncClient",
            return_value=client,
        ) as factory:
            elevenlabs_tts._default_client = None
            elevenlabs_tts._default_client_loop = None
            await warmup_elevenlabs_connection(api_key="api-key")
            await warmup_elevenlabs_connection(api_key="api-key")

        factory.assert_called_once()
        self.assertEqual(client.get.await_count, 2)
        client.post.assert_not_awaited()

    async def test_synthesizes_with_explicit_member_voice(self) -> None:
        client = _Client(_Response(200, b"member-audio"))

        result = await synthesize_member_speech(
            text="안녕하세요.",
            voice_id="member-voice",
            api_key="api-key",
            settings=ElevenLabsVoiceSettings(similarity_boost=0.9),
            http_client=client,
        )

        self.assertEqual(result, b"member-audio")
        url, request = client.request
        self.assertTrue(url.endswith("/member-voice"))
        self.assertEqual(request["json"]["text"], "안녕하세요.")
        self.assertEqual(request["json"]["model_id"], "eleven_multilingual_v2")
        self.assertEqual(request["params"]["output_format"], "mp3_44100_128")

    async def test_uses_explicit_model_without_changing_voice(self) -> None:
        client = _Client(_Response(200, b"member-audio"))

        await synthesize_member_speech(
            text="안녕하세요.",
            voice_id="member-voice",
            api_key="api-key",
            model_id="eleven_flash_v2_5",
            http_client=client,
        )

        url, request = client.request
        self.assertTrue(url.endswith("/member-voice"))
        self.assertEqual(request["json"]["model_id"], "eleven_flash_v2_5")

    async def test_does_not_include_remote_error_body(self) -> None:
        client = _Client(_Response(403, b"sensitive upstream response"))

        with self.assertRaisesRegex(ElevenLabsTTSError, "HTTP 403") as raised:
            await synthesize_member_speech(
                text="안녕하세요.",
                voice_id="member-voice",
                api_key="api-key",
                http_client=client,
            )

        self.assertNotIn("sensitive", str(raised.exception))

    async def test_reuses_default_http_client_on_same_event_loop(self) -> None:
        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=_Response(200, b"audio"))
        with patch("shared.elevenlabs_tts.httpx.AsyncClient", return_value=client) as factory:
            elevenlabs_tts._default_client = None
            elevenlabs_tts._default_client_loop = None
            await synthesize_member_speech(
                text="첫 번째",
                voice_id="member-voice",
                api_key="api-key",
            )
            await synthesize_member_speech(
                text="두 번째",
                voice_id="member-voice",
                api_key="api-key",
            )

        factory.assert_called_once()
        self.assertEqual(client.post.await_count, 2)


if __name__ == "__main__":
    unittest.main()
