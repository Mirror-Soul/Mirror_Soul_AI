from __future__ import annotations

import asyncio
import hmac
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from ditto_server.config import DittoServiceConfig
from ditto_server.engine import (
    DittoEngine,
    DittoEngineBusyError,
    DittoEngineError,
)
from model_training.face_training.ditto_runner import (
    DittoRenderSettings,
    DittoRunnerError,
    load_ditto_render_settings,
    validate_ditto_render_settings,
)


PORTRAIT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac"}
PROFILE_MAX_BYTES = 128 * 1024


def create_app(
    config: DittoServiceConfig,
    *,
    engine: DittoEngine | None = None,
) -> FastAPI:
    render_engine = engine or DittoEngine(config)
    # CUDA and ONNX sessions must load and render on the same OS thread.
    gpu_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ditto-gpu",
    )
    render_guard = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(gpu_executor, render_engine.load)
            app.state.ditto_engine = render_engine
            yield
        finally:
            gpu_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="Mirror Soul Ditto GPU Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "ditto-gpu",
            "engine": render_engine.status(),
        }

    @app.get("/ready")
    async def ready():
        status = render_engine.status()
        if not status.get("loaded"):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not-ready",
                    "service": "ditto-gpu",
                    "engine": status,
                },
            )
        return {
            "status": "ready",
            "service": "ditto-gpu",
            "engine": status,
        }

    @app.post("/api/v1/render")
    async def render(
        portrait: UploadFile = File(...),
        audio: UploadFile = File(...),
        profile: UploadFile | None = File(None),
        crop_scale: float | None = Form(None),
        smoothing_kernel: int | None = Form(None),
        sampling_timesteps: int | None = Form(None),
        seed: int = Form(1024),
        x_ditto_api_key: str | None = Header(
            None,
            alias="X-Ditto-Api-Key",
        ),
    ):
        _require_api_key(x_ditto_api_key, config.api_key)
        if render_guard.locked():
            raise HTTPException(
                status_code=429,
                detail="Ditto GPU is already rendering.",
            )

        async with render_guard:
            return await _render_response(
                render_engine,
                gpu_executor,
                config,
                portrait=portrait,
                audio=audio,
                profile=profile,
                crop_scale=crop_scale,
                smoothing_kernel=smoothing_kernel,
                sampling_timesteps=sampling_timesteps,
                seed=seed,
            )

    return app


async def _render_response(
    render_engine: DittoEngine,
    gpu_executor: ThreadPoolExecutor,
    config: DittoServiceConfig,
    *,
    portrait: UploadFile,
    audio: UploadFile,
    profile: UploadFile | None,
    crop_scale: float | None,
    smoothing_kernel: int | None,
    sampling_timesteps: int | None,
    seed: int,
):
    request_id = uuid4().hex
    workspace = tempfile.TemporaryDirectory(
        prefix=f"mirror-soul-ditto-{request_id}-"
    )
    workspace_path = Path(workspace.name)
    try:
        portrait_path = workspace_path / (
            "portrait" + _validated_suffix(portrait, PORTRAIT_SUFFIXES)
        )
        audio_path = workspace_path / (
            "audio" + _validated_suffix(audio, AUDIO_SUFFIXES)
        )
        await _save_upload(
            portrait,
            portrait_path,
            max_bytes=config.max_portrait_bytes,
        )
        await _save_upload(
            audio,
            audio_path,
            max_bytes=config.max_audio_bytes,
        )

        settings = DittoRenderSettings()
        if profile is not None:
            profile_path = workspace_path / "face-profile.json"
            await _save_upload(
                profile,
                profile_path,
                max_bytes=PROFILE_MAX_BYTES,
            )
            settings = load_ditto_render_settings(profile_path)
        settings = _apply_overrides(
            settings,
            crop_scale=crop_scale,
            smoothing_kernel=smoothing_kernel,
            sampling_timesteps=sampling_timesteps,
        )
        output_path = workspace_path / "render.mp4"
        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            gpu_executor,
            partial(
                render_engine.render,
                portrait_path,
                audio_path,
                output_path,
                settings=settings,
                seed=seed,
            ),
        )
    except HTTPException:
        workspace.cleanup()
        raise
    except DittoEngineBusyError as exc:
        workspace.cleanup()
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except DittoRunnerError as exc:
        workspace.cleanup()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DittoEngineError as exc:
        workspace.cleanup()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        workspace.cleanup()
        raise

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"ditto-{request_id}.mp4",
        headers={
            "X-Ditto-Request-Id": request_id,
            "X-Ditto-Render-Seconds": f"{metrics.duration_seconds:.3f}",
            "X-Ditto-Output-Bytes": str(metrics.output_size_bytes),
        },
        background=BackgroundTask(workspace.cleanup),
    )


def _require_api_key(provided: str | None, expected: str) -> None:
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid Ditto API key.")


def _validated_suffix(upload: UploadFile, allowed: set[str]) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        values = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported upload extension. Allowed: {values}",
        )
    return suffix


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
) -> None:
    size_bytes = 0
    try:
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {max_bytes} bytes.",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Upload must not be empty.")


def _apply_overrides(
    settings: DittoRenderSettings,
    *,
    crop_scale: float | None,
    smoothing_kernel: int | None,
    sampling_timesteps: int | None,
) -> DittoRenderSettings:
    if crop_scale is not None:
        settings = replace(settings, crop_scale=crop_scale)
    if smoothing_kernel is not None:
        settings = replace(settings, smoothing_kernel=smoothing_kernel)
    if sampling_timesteps is not None:
        settings = replace(
            settings,
            sampling_timesteps=sampling_timesteps,
        )
    validate_ditto_render_settings(settings)
    return settings
