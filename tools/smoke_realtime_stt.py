"""Open and close a Realtime transcription session without sending audio."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import av

from model_calling.realtime.transcription import RealtimeTranscriber
from shared.config import settings


async def run(audio_path: Path | None) -> None:
    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not configured")
    transcriber = await RealtimeTranscriber.connect(
        api_key=settings.OPENAI_API_KEY,
        connection_model=settings.REALTIME_STREAMING_CONNECTION_MODEL,
        model=settings.REALTIME_STREAMING_STT_MODEL,
        language=settings.STT_LANGUAGE,
        timeout_seconds=settings.REALTIME_STREAMING_STT_CONNECT_TIMEOUT_SECONDS,
    )
    try:
        print(
            "Realtime STT connection OK: "
            f"connection={settings.REALTIME_STREAMING_CONNECTION_MODEL} "
            f"transcription={settings.REALTIME_STREAMING_STT_MODEL}"
        )
        if audio_path is not None:
            container = av.open(str(audio_path))
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=24000,
            )
            try:
                for frame in container.decode(audio=0):
                    for output in resampler.resample(frame):
                        await transcriber.append_audio(output.to_ndarray().tobytes())
                for output in resampler.resample(None):
                    await transcriber.append_audio(output.to_ndarray().tobytes())
            finally:
                container.close()
            future = await transcriber.commit()
            transcript = await asyncio.wait_for(future, timeout=15)
            print(f"Transcript: {transcript}")
    finally:
        await transcriber.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    if args.audio is not None and not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")
    asyncio.run(run(args.audio))


if __name__ == "__main__":
    main()
