import os
from dataclasses import dataclass
from typing import Any


class ElevenLabsTTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class ElevenLabsVoiceSettings:
    stability: float = 0.5
    similarity_boost: float = 0.9
    style: float = 0.0
    use_speaker_boost: bool = True

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "stability": _clamp(self.stability),
            "similarity_boost": _clamp(self.similarity_boost),
            "style": _clamp(self.style),
            "use_speaker_boost": self.use_speaker_boost,
        }


async def synthesize_member_speech(
    *,
    text: str,
    voice_id: str,
    settings: ElevenLabsVoiceSettings | None = None,
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    timeout_seconds: float = 60.0,
    http_client: Any | None = None,
) -> bytes:
    normalized_text = text.strip()
    normalized_voice_id = voice_id.strip()
    resolved_api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not normalized_text:
        raise ValueError("text must not be empty")
    if not normalized_voice_id:
        raise ValueError("voice_id must not be empty")
    if not resolved_api_key:
        raise ElevenLabsTTSError("ELEVENLABS_API_KEY is not configured")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{normalized_voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": resolved_api_key,
    }
    payload = {
        "text": normalized_text,
        "model_id": model_id,
        "voice_settings": (
            settings or ElevenLabsVoiceSettings()
        ).to_dict(),
    }
    params = {"output_format": output_format}

    if http_client is not None:
        response = await http_client.post(
            url,
            headers=headers,
            json=payload,
            params=params,
        )
    else:
        try:
            import httpx
        except ImportError as exc:
            raise ElevenLabsTTSError(
                "httpx is required for ElevenLabs TTS requests"
            ) from exc
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                params=params,
            )

    if response.status_code != 200:
        raise ElevenLabsTTSError(
            f"ElevenLabs TTS failed with HTTP {response.status_code}"
        )
    if not response.content:
        raise ElevenLabsTTSError("ElevenLabs TTS returned empty audio")
    return bytes(response.content)


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
