from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import hmac
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from ditto_server.config import DittoServiceConfig
from ditto_server.engine import (
    DEFAULT_FADE_KEYS,
    DittoEngine,
    DittoEngineBusyError,
    DittoEngineError,
    DittoMotionOptions,
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
STREAM_CONTENT_TYPE = "application/vnd.mirrorsoul.ditto-frame-stream"
STREAM_MAGIC = b"MSDS1\n"


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
        fade_in_frames: int | None = Form(None),
        fade_out_frames: int | None = Form(None),
        fade_type: str | None = Form(None),
        fade_keys: str | None = Form(None),
        x_ditto_api_key: str | None = Header(
            None,
            alias="X-Ditto-Api-Key",
        ),
    ):
        _require_api_key(x_ditto_api_key, config.api_key)
        motion = _motion_options(
            fade_in_frames=fade_in_frames,
            fade_out_frames=fade_out_frames,
            fade_type=fade_type,
            fade_keys=fade_keys,
        )
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
                motion=motion,
            )

    @app.post("/api/v1/render/stream")
    async def render_stream(
        portrait: UploadFile = File(...),
        audio: UploadFile = File(...),
        profile: UploadFile | None = File(None),
        crop_scale: float | None = Form(None),
        smoothing_kernel: int | None = Form(None),
        sampling_timesteps: int | None = Form(None),
        seed: int = Form(1024),
        fade_in_frames: int | None = Form(None),
        fade_out_frames: int | None = Form(None),
        fade_type: str | None = Form(None),
        fade_keys: str | None = Form(None),
        max_width: int | None = Form(None),
        max_height: int | None = Form(None),
        x_ditto_api_key: str | None = Header(
            None,
            alias="X-Ditto-Api-Key",
        ),
    ):
        _require_api_key(x_ditto_api_key, config.api_key)
        motion = _motion_options(
            fade_in_frames=fade_in_frames,
            fade_out_frames=fade_out_frames,
            fade_type=fade_type,
            fade_keys=fade_keys,
        )
        max_size = _stream_max_size(max_width, max_height)
        status = render_engine.status()
        if not config.streaming_enabled or not status.get("streamingAvailable"):
            raise HTTPException(
                status_code=503,
                detail="Ditto online streaming is unavailable.",
            )
        if render_guard.locked():
            raise HTTPException(
                status_code=429,
                detail="Ditto GPU is already rendering.",
            )

        workspace, portrait_path, audio_path, settings = await _prepare_inputs(
            config,
            portrait=portrait,
            audio=audio,
            profile=profile,
            crop_scale=crop_scale,
            smoothing_kernel=smoothing_kernel,
            sampling_timesteps=sampling_timesteps,
        )
        await render_guard.acquire()
        return _streaming_response(
            render_engine,
            gpu_executor,
            render_guard,
            config,
            workspace=workspace,
            portrait_path=portrait_path,
            audio_path=audio_path,
            settings=settings,
            seed=seed,
            motion=motion,
            max_size=max_size,
        )

    return app


def _stream_max_size(
    max_width: int | None,
    max_height: int | None,
) -> tuple[int, int] | None:
    """Optional bound for streamed frames (the call server's output size)."""
    if max_width is None and max_height is None:
        return None
    if max_width is None or max_height is None:
        raise HTTPException(
            status_code=422,
            detail="max_width and max_height must be sent together.",
        )
    if not (16 <= max_width <= 4096 and 16 <= max_height <= 4096):
        raise HTTPException(
            status_code=422,
            detail="max_width and max_height must be between 16 and 4096.",
        )
    return max_width, max_height


async def _prepare_inputs(
    config: DittoServiceConfig,
    *,
    portrait: UploadFile,
    audio: UploadFile,
    profile: UploadFile | None,
    crop_scale: float | None,
    smoothing_kernel: int | None,
    sampling_timesteps: int | None,
) -> tuple[tempfile.TemporaryDirectory, Path, Path, DittoRenderSettings]:
    workspace = tempfile.TemporaryDirectory(prefix="mirror-soul-ditto-stream-")
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
        await _save_upload(audio, audio_path, max_bytes=config.max_audio_bytes)
        settings = DittoRenderSettings()
        if profile is not None:
            profile_path = workspace_path / "face-profile.json"
            await _save_upload(profile, profile_path, max_bytes=PROFILE_MAX_BYTES)
            settings = load_ditto_render_settings(profile_path)
        settings = _apply_overrides(
            settings,
            crop_scale=crop_scale,
            smoothing_kernel=smoothing_kernel,
            sampling_timesteps=sampling_timesteps,
        )
        return workspace, portrait_path, audio_path, settings
    except Exception:
        workspace.cleanup()
        raise


def _streaming_response(
    render_engine: DittoEngine,
    gpu_executor: ThreadPoolExecutor,
    render_guard: asyncio.Lock,
    config: DittoServiceConfig,
    *,
    workspace: tempfile.TemporaryDirectory,
    portrait_path: Path,
    audio_path: Path,
    settings: DittoRenderSettings,
    seed: int,
    motion: DittoMotionOptions | None = None,
    max_size: tuple[int, int] | None = None,
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(maxsize=8)
    stopped = threading.Event()
    output_path = Path(workspace.name) / "stream.mp4"

    def put_from_worker(item: bytes | Exception | None) -> bool:
        while not stopped.is_set():
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                future.result(timeout=0.25)
                return True
            except FutureTimeoutError:
                future.cancel()
        return False

    def publish(frame) -> None:
        packet = _encode_stream_frame(
            frame,
            config.stream_jpeg_quality,
            max_size=max_size,
        )
        if not put_from_worker(packet):
            raise DittoEngineError("Ditto stream client disconnected.")

    stream_kwargs: dict[str, object] = {}
    if motion is not None:
        # Only sent when requested so engines without fade support still work.
        stream_kwargs["motion"] = motion

    def run() -> None:
        try:
            render_engine.stream_frames(
                portrait_path,
                audio_path,
                output_path,
                settings=settings,
                on_frame=publish,
                seed=seed,
                should_cancel=stopped.is_set,
                **stream_kwargs,
            )
        except Exception as exc:
            put_from_worker(exc)
        finally:
            put_from_worker(None)

    worker = loop.run_in_executor(gpu_executor, run)

    async def body():
        try:
            yield STREAM_MAGIC
            while True:
                item = await queue.get()
                if item is None:
                    yield struct.pack(">I", 0)
                    break
                if isinstance(item, Exception):
                    raise item
                yield struct.pack(">I", len(item)) + item
        finally:
            stopped.set()
            try:
                await worker
            finally:
                workspace.cleanup()
                if render_guard.locked():
                    render_guard.release()

    return StreamingResponse(
        body(),
        media_type=STREAM_CONTENT_TYPE,
        headers={"X-Ditto-Stream-Version": "1"},
    )


def _encode_stream_frame(
    frame,
    quality: int,
    *,
    max_size: tuple[int, int] | None = None,
) -> bytes:
    """JPEG-encode one BGR frame, shrunk to fit ``max_size`` if given.

    The call server shows 540x960 by default while Ditto renders larger
    frames; shrinking here cuts the bytes sent over the tunnel and the
    decoding work on the call server.
    """
    import cv2

    if max_size is not None:
        height, width = frame.shape[:2]
        scale = min(max_size[0] / width, max_size[1] / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise DittoEngineError("Unable to encode Ditto stream frame.")
    return encoded.tobytes()


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
    motion: DittoMotionOptions | None = None,
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
        render_kwargs: dict[str, object] = {"settings": settings, "seed": seed}
        if motion is not None:
            render_kwargs["motion"] = motion
        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            gpu_executor,
            partial(
                render_engine.render,
                portrait_path,
                audio_path,
                output_path,
                **render_kwargs,
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


def _motion_options(
    *,
    fade_in_frames: int | None,
    fade_out_frames: int | None,
    fade_type: str | None,
    fade_keys: str | None,
) -> DittoMotionOptions | None:
    if fade_in_frames is None and fade_out_frames is None:
        return None
    keys = DEFAULT_FADE_KEYS
    if fade_keys is not None and fade_keys.strip():
        keys = tuple(
            key.strip() for key in fade_keys.split(",") if key.strip()
        )
    motion = DittoMotionOptions(
        fade_in_frames=fade_in_frames or 0,
        fade_out_frames=fade_out_frames or 0,
        fade_type=(fade_type or "s").strip(),
        fade_keys=keys,
    )
    try:
        motion.validate()
    except DittoEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return motion if motion.active else None


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
