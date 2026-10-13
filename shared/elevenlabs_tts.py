import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any

import httpx


_default_client: httpx.AsyncClient | None = None
_default_client_loop: asyncio.AbstractEventLoop | None = None
_default_client_lock = threading.Lock()


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


async def warmup_elevenlabs_connection(
    *,
    api_key: str | None = None,
    timeout_seconds: float = 5.0,
) -> None:
    """Prime the pooled ElevenLabs connection without generating audio."""
    resolved_api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not resolved_api_key:
        raise ElevenLabsTTSError("ELEVENLABS_API_KEY is not configured")
    response = await _default_http_client(timeout_seconds).get(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": resolved_api_key},
        timeout=timeout_seconds,
    )
    if response.status_code != 200:
        raise ElevenLabsTTSError(
            f"ElevenLabs warm-up failed with HTTP {response.status_code}"
        )


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
        response = await _default_http_client(timeout_seconds).post(
            url,
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout_seconds,
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


def _default_http_client(timeout_seconds: float) -> httpx.AsyncClient:
    """Reuse TLS connections while keeping clients isolated by event loop."""
    global _default_client, _default_client_loop

    loop = asyncio.get_running_loop()
    with _default_client_lock:
        if (
            _default_client is None
            or _default_client.is_closed
            or _default_client_loop is not loop
        ):
            _default_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
            )
            _default_client_loop = loop
        return _default_client
