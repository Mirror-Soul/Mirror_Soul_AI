from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import struct
import subprocess
import time
import wave
from pathlib import Path

import httpx


STREAM_MAGIC = b"MSDS1\n"


class _FrameParser:
    def __init__(self, max_frame_bytes: int = 5 * 1024 * 1024) -> None:
        self.max_frame_bytes = max_frame_bytes
        self.buffer = bytearray()
        self.magic_read = False
        self.completed = False

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        frames: list[bytes] = []
        if not self.magic_read:
            if len(self.buffer) < len(STREAM_MAGIC):
                return frames
            if bytes(self.buffer[: len(STREAM_MAGIC)]) != STREAM_MAGIC:
                raise ValueError("Invalid Ditto stream header.")
            del self.buffer[: len(STREAM_MAGIC)]
            self.magic_read = True
        while len(self.buffer) >= 4 and not self.completed:
            size = struct.unpack(">I", self.buffer[:4])[0]
            if size == 0:
                del self.buffer[:4]
                self.completed = True
                break
            if size > self.max_frame_bytes:
                raise ValueError(f"Frame exceeds {self.max_frame_bytes} bytes.")
            if len(self.buffer) < 4 + size:
                break
            frames.append(bytes(self.buffer[4 : 4 + size]))
            del self.buffer[: 4 + size]
        return frames

    def finish(self) -> None:
        if not self.magic_read or not self.completed or self.buffer:
            raise ValueError("Ditto stream ended unexpectedly.")


def _audio_duration(audio: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except (EOFError, wave.Error, ZeroDivisionError):
        return None


def _stream_quality(first_frame_ms: int, generation_fps: float) -> str:
    if first_frame_ms >= 3000 or generation_fps < 20:
        return "FAIL"
    if first_frame_ms >= 1500 or generation_fps < 24:
        return "WARNING"
    return "PASS"


async def _render(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    portrait: bytes,
    portrait_name: str,
    audio: bytes,
    audio_name: str,
    profile: bytes | None,
) -> dict[str, object]:
    files = {
        "portrait": (portrait_name, portrait, "image/jpeg"),
        "audio": (audio_name, audio, "audio/wav"),
    }
    if profile is not None:
        files["profile"] = (
            "face-profile.json",
            profile,
            "application/json",
        )
    started = time.monotonic()
    response = await client.post(
        f"{url.rstrip('/')}/api/v1/render",
        headers={"X-Ditto-Api-Key": api_key},
        files=files,
    )
    elapsed = time.monotonic() - started
    return {
        "url": url,
        "status": response.status_code,
        "wallSeconds": round(elapsed, 3),
        "renderSeconds": response.headers.get("x-ditto-render-seconds"),
        "outputBytes": len(response.content),
        "contentType": response.headers.get("content-type"),
    }


async def _stream(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    portrait: bytes,
    portrait_name: str,
    audio: bytes,
    audio_name: str,
    profile: bytes | None,
    *,
    sampling_timesteps: int,
    max_width: int,
    max_height: int,
    consumer_delay_ms: float,
) -> dict[str, object]:
    files = {
        "portrait": (portrait_name, portrait, "image/jpeg"),
        "audio": (audio_name, audio, "audio/wav"),
    }
    if profile is not None:
        files["profile"] = (
            "face-profile.json",
            profile,
            "application/json",
        )
    data = {"sampling_timesteps": str(sampling_timesteps)}
    if max_width > 0 and max_height > 0:
        data.update({"max_width": str(max_width), "max_height": str(max_height)})

    started = time.monotonic()
    first_frame_at: float | None = None
    frame_count = 0
    total_bytes = 0
    parser = _FrameParser()
    async with client.stream(
        "POST",
        f"{url.rstrip('/')}/api/v1/render/stream",
        headers={"X-Ditto-Api-Key": api_key},
        files=files,
        data=data,
    ) as response:
        if response.status_code != 200:
            content = await response.aread()
            return {
                "url": url,
                "status": response.status_code,
                "error": content.decode("utf-8", errors="replace")[:500],
            }
        async for chunk in response.aiter_bytes():
            for frame in parser.feed(chunk):
                if first_frame_at is None:
                    first_frame_at = time.monotonic()
                frame_count += 1
                total_bytes += len(frame)
                if consumer_delay_ms > 0:
                    await asyncio.sleep(consumer_delay_ms / 1000)
        parser.finish()

    finished = time.monotonic()
    first_frame_ms = round(((first_frame_at or finished) - started) * 1000)
    generation_seconds = max(0.001, finished - (first_frame_at or finished))
    generation_fps = max(0, frame_count - 1) / generation_seconds
    wall_seconds = finished - started
    duration = _audio_duration(audio)
    return {
        "url": url,
        "status": 200,
        "quality": _stream_quality(first_frame_ms, generation_fps),
        "firstFrameMs": first_frame_ms,
        "wallSeconds": round(wall_seconds, 3),
        "audioSeconds": round(duration, 3) if duration is not None else None,
        "frameCount": frame_count,
        "generationFps": round(generation_fps, 2),
        "realtimeFactor": (
            round(duration / wall_seconds, 3)
            if duration is not None and wall_seconds > 0
            else None
        ),
        "streamMbps": round(
            total_bytes * 8 / max(duration or wall_seconds, 0.001) / 1_000_000,
            2,
        ),
        "consumerDelayMs": consumer_delay_ms,
    }


def _gpu_sample() -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization = completed.stdout.strip().split(",", maxsplit=1)
    return int(memory.strip()), int(utilization.strip())


async def _sample_gpu(tasks: list[asyncio.Task]) -> dict[str, int]:
    peak_memory = 0
    peak_utilization = 0
    while not all(task.done() for task in tasks):
        memory, utilization = await asyncio.to_thread(_gpu_sample)
        peak_memory = max(peak_memory, memory)
        peak_utilization = max(peak_utilization, utilization)
        await asyncio.sleep(0.2)
    memory, utilization = await asyncio.to_thread(_gpu_sample)
    return {
        "peakMemoryMiB": max(peak_memory, memory),
        "peakUtilizationPercent": max(peak_utilization, utilization),
    }


async def _run(args: argparse.Namespace) -> dict[str, object]:
    urls = tuple(item.strip() for item in args.urls.split(",") if item.strip())
    if not urls:
        raise ValueError("At least one worker URL is required.")
    api_key = os.getenv("DITTO_SERVICE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DITTO_SERVICE_API_KEY is not set.")

    portrait_path = Path(args.portrait)
    audio_path = Path(args.audio)
    profile_path = Path(args.profile) if args.profile else None
    portrait = portrait_path.read_bytes()
    audio = audio_path.read_bytes()
    profile = profile_path.read_bytes() if profile_path else None

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = []
        for url in urls:
            if args.mode == "stream":
                coroutine = _stream(
                    client,
                    url,
                    api_key,
                    portrait,
                    portrait_path.name,
                    audio,
                    audio_path.name,
                    profile,
                    sampling_timesteps=args.sampling_timesteps,
                    max_width=args.max_width,
                    max_height=args.max_height,
                    consumer_delay_ms=args.consumer_delay_ms,
                )
            else:
                coroutine = _render(
                    client,
                    url,
                    api_key,
                    portrait,
                    portrait_path.name,
                    audio,
                    audio_path.name,
                    profile,
                )
            tasks.append(asyncio.create_task(coroutine))
        sampler = asyncio.create_task(_sample_gpu(tasks))
        started = time.monotonic()
        results = await asyncio.gather(*tasks)
        gpu = await sampler

    return {
        "mode": args.mode,
        "workerCount": len(urls),
        "totalWallSeconds": round(time.monotonic() - started, 3),
        **gpu,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent Ditto worker processes.",
    )
    parser.add_argument("--urls", required=True)
    parser.add_argument("--portrait", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--mode", choices=("render", "stream"), default="stream")
    parser.add_argument("--sampling-timesteps", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=540)
    parser.add_argument("--max-height", type=int, default=960)
    parser.add_argument(
        "--consumer-delay-ms",
        type=float,
        default=0.0,
        help="artificial delay after each received frame",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if any(item["status"] != 200 for item in result["results"]):
        raise SystemExit(1)
    if any(item.get("quality") == "FAIL" for item in result["results"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
