from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx


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
        tasks = [
            asyncio.create_task(
                _render(
                    client,
                    url,
                    api_key,
                    portrait,
                    portrait_path.name,
                    audio,
                    audio_path.name,
                    profile,
                )
            )
            for url in urls
        ]
        sampler = asyncio.create_task(_sample_gpu(tasks))
        started = time.monotonic()
        results = await asyncio.gather(*tasks)
        gpu = await sampler

    return {
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
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if any(item["status"] != 200 for item in result["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
