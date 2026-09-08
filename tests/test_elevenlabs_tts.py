import unittest

from shared.elevenlabs_tts import (
    ElevenLabsTTSError,
    ElevenLabsVoiceSettings,
    synthesize_member_speech,
)


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
        self.assertEqual(request["params"]["output_format"], "mp3_44100_128")

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


if __name__ == "__main__":
    unittest.main()
