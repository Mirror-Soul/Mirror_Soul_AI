from __future__ import annotations

import importlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from ditto_server.config import DittoServiceConfig
from model_training.face_training.ditto_runner import (
    DittoRenderSettings,
    validate_ditto_render_settings,
)


class DittoEngineError(RuntimeError):
    pass


class DittoEngineBusyError(DittoEngineError):
    pass


@dataclass(frozen=True)
class DittoRuntime:
    sdk_factory: Callable[[str, str], Any]
    run: Callable[..., None]
    seed_everything: Callable[[int], None]
    cuda_available: Callable[[], bool]
    gpu_name: Callable[[], str]


@dataclass(frozen=True)
class DittoRenderMetrics:
    duration_seconds: float
    output_size_bytes: int


class DittoEngine:
    def __init__(
        self,
        config: DittoServiceConfig,
        *,
        runtime_loader: Callable[[Path], DittoRuntime] | None = None,
    ) -> None:
        self.config = config
        self._runtime_loader = runtime_loader or _load_runtime
        self._runtime: DittoRuntime | None = None
        self._sdk: Any = None
        self._render_slot = threading.Lock()
        self._state_lock = threading.Lock()
        self._loaded = False
        self._busy = False
        self._load_seconds: float | None = None
        self._render_count = 0
        self._last_render_seconds: float | None = None
        self._last_error: str | None = None
        self._gpu_name: str | None = None

    def load(self) -> None:
        with self._state_lock:
            if self._loaded:
                return

        self.config.validate()
        repository_dir = self.config.repository_dir.resolve()
        _prepend_path(self.config.ffmpeg_dir.resolve())
        started = monotonic()
        try:
            runtime = self._runtime_loader(repository_dir)
            if self.config.require_cuda and not runtime.cuda_available():
                raise DittoEngineError("CUDA is required but unavailable.")
            sdk = runtime.sdk_factory(
                str(self.config.resolved_model_config_path),
                str(self.config.resolved_data_root),
            )
            gpu_name = runtime.gpu_name() if runtime.cuda_available() else "none"
        except Exception as exc:
            with self._state_lock:
                self._last_error = str(exc)
            if isinstance(exc, DittoEngineError):
                raise
            raise DittoEngineError(f"Ditto model load failed: {exc}") from exc

        with self._state_lock:
            self._runtime = runtime
            self._sdk = sdk
            self._gpu_name = gpu_name
            self._load_seconds = monotonic() - started
            self._loaded = True
            self._last_error = None

    def render(
        self,
        source_path: Path,
        audio_path: Path,
        output_path: Path,
        *,
        settings: DittoRenderSettings,
        seed: int = 1024,
    ) -> DittoRenderMetrics:
        validate_ditto_render_settings(settings)
        source_path = source_path.resolve()
        audio_path = audio_path.resolve()
        output_path = output_path.resolve()
        if not source_path.is_file():
            raise DittoEngineError(f"Source image not found: {source_path}")
        if not audio_path.is_file():
            raise DittoEngineError(f"Audio file not found: {audio_path}")
        if output_path.suffix.lower() != ".mp4":
            raise DittoEngineError("Ditto output must use the .mp4 extension.")

        with self._state_lock:
            if not self._loaded or self._runtime is None or self._sdk is None:
                raise DittoEngineError("Ditto model is not loaded.")

        if not self._render_slot.acquire(blocking=False):
            raise DittoEngineBusyError("Ditto GPU is already rendering.")

        temporary_video_path = Path(str(output_path) + ".tmp.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        temporary_video_path.unlink(missing_ok=True)
        with self._state_lock:
            self._busy = True
            self._last_error = None

        started = monotonic()
        try:
            self._runtime.seed_everything(seed)
            self._runtime.run(
                self._sdk,
                str(audio_path),
                str(source_path),
                str(output_path),
                more_kwargs={
                    "setup_kwargs": {
                        "crop_scale": settings.crop_scale,
                        "smo_k_d": settings.smoothing_kernel,
                        "sampling_timesteps": settings.sampling_timesteps,
                    }
                },
            )
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise DittoEngineError("Ditto did not produce a final MP4 file.")
            duration_seconds = monotonic() - started
            metrics = DittoRenderMetrics(
                duration_seconds=duration_seconds,
                output_size_bytes=output_path.stat().st_size,
            )
            with self._state_lock:
                self._render_count += 1
                self._last_render_seconds = duration_seconds
                self._last_error = None
            return metrics
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            temporary_video_path.unlink(missing_ok=True)
            with self._state_lock:
                self._last_error = str(exc)
            if isinstance(exc, DittoEngineError):
                raise
            raise DittoEngineError(f"Ditto render failed: {exc}") from exc
        finally:
            with self._state_lock:
                self._busy = False
            self._render_slot.release()

    def status(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "loaded": self._loaded,
                "busy": self._busy,
                "gpu": self._gpu_name,
                "modelLoadSeconds": self._load_seconds,
                "renderCount": self._render_count,
                "lastRenderSeconds": self._last_render_seconds,
                "lastError": self._last_error,
            }


def _load_runtime(repository_dir: Path) -> DittoRuntime:
    repository = str(repository_dir)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    os.chdir(repository_dir)
    inference = importlib.import_module("inference")
    pipeline = importlib.import_module("stream_pipeline_offline")
    torch = importlib.import_module("torch")
    return DittoRuntime(
        sdk_factory=pipeline.StreamSDK,
        run=inference.run,
        seed_everything=inference.seed_everything,
        cuda_available=torch.cuda.is_available,
        gpu_name=lambda: torch.cuda.get_device_name(0),
    )


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    value = str(path)
    if value not in entries:
        os.environ["PATH"] = os.pathsep.join((value, current))
