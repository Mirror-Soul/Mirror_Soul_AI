from __future__ import annotations

import importlib
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import numpy as np

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
    online_sdk_factory: Callable[[str, str], Any] | None = None
    load_audio: Callable[[str], np.ndarray] | None = None


FADE_TYPES = frozenset({"s", "d0"})
FADE_KEYS = frozenset({"exp", "pitch", "yaw", "roll", "t", "scale"})
DEFAULT_FADE_KEYS = ("exp", "pitch", "yaw", "roll", "t")
MAX_FADE_FRAMES = 250
NUMPY_COMPAT_ALIASES = {
    "atan2": "arctan2",
    "atan": "arctan",
    "asin": "arcsin",
    "acos": "arccos",
    "pow": "power",
    "concat": "concatenate",
}


@dataclass(frozen=True)
class DittoMotionOptions:
    """Optional motion controls passed to Ditto for one render.

    ``fade_type="s"`` blends the first ``fade_in_frames`` and the last
    ``fade_out_frames`` toward the source portrait's own pose and
    expression, so a clip starts and ends on the same neutral frame. That
    lets the call server join clips and idle footage without visible jumps.
    """

    fade_in_frames: int = 0
    fade_out_frames: int = 0
    fade_type: str = "s"
    fade_keys: tuple[str, ...] = DEFAULT_FADE_KEYS

    @property
    def active(self) -> bool:
        return self.fade_in_frames > 0 or self.fade_out_frames > 0

    def validate(self) -> None:
        for value in (self.fade_in_frames, self.fade_out_frames):
            if value < 0 or value > MAX_FADE_FRAMES:
                raise DittoEngineError(
                    f"Ditto fade frames must be between 0 and {MAX_FADE_FRAMES}."
                )
        if self.fade_type not in FADE_TYPES:
            raise DittoEngineError("Ditto fade type must be 's' or 'd0'.")
        if not self.fade_keys or not set(self.fade_keys) <= FADE_KEYS:
            raise DittoEngineError(
                "Ditto fade keys must be a non-empty subset of "
                f"{sorted(FADE_KEYS)}."
            )


def build_ditto_more_kwargs(
    settings: DittoRenderSettings,
    motion: DittoMotionOptions | None = None,
) -> dict[str, dict[str, Any]]:
    setup_kwargs: dict[str, Any] = {
        "crop_scale": settings.crop_scale,
        "smo_k_d": settings.smoothing_kernel,
        "sampling_timesteps": settings.sampling_timesteps,
    }
    more_kwargs: dict[str, dict[str, Any]] = {"setup_kwargs": setup_kwargs}
    if motion is not None and motion.active:
        setup_kwargs["fade_type"] = motion.fade_type
        setup_kwargs["fade_out_keys"] = tuple(motion.fade_keys)
        more_kwargs["run_kwargs"] = {
            "fade_in": motion.fade_in_frames or -1,
            "fade_out": motion.fade_out_frames or -1,
            # A fresh dict per render: Ditto mutates ctrl_info in place.
            "ctrl_info": {},
        }
    return more_kwargs


@dataclass(frozen=True)
class DittoRenderMetrics:
    duration_seconds: float
    output_size_bytes: int


@dataclass(frozen=True)
class DittoStreamMetrics:
    duration_seconds: float
    frame_count: int


class _FrameCallbackWriter:
    """Forwards every frame Ditto writes to ``callback`` in BGR order.

    Ditto's online pipeline calls its writer with ``fmt="rgb"`` frames, while
    OpenCV (used to JPEG-encode stream frames) expects BGR. Passing RGB
    straight to ``cv2.imencode`` swaps red and blue in every streamed frame.
    """

    def __init__(
        self,
        writer: Any,
        callback: Callable[[np.ndarray], None],
    ) -> None:
        self._writer = writer
        self._callback = callback

    def __call__(self, frame: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        fmt = kwargs.get("fmt", args[0] if args else "bgr")
        bgr = frame[..., ::-1] if str(fmt).lower() == "rgb" else frame
        self._callback(np.ascontiguousarray(bgr))
        return self._writer(frame, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


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
        self._online_sdk: Any = None
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
            online_sdk = None
            online_config_path = self.config.resolved_online_model_config_path
            if (
                self.config.streaming_enabled
                and runtime.online_sdk_factory is not None
                and runtime.load_audio is not None
                and online_config_path.is_file()
            ):
                online_sdk = runtime.online_sdk_factory(
                    str(online_config_path),
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
            self._online_sdk = online_sdk
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
        motion: DittoMotionOptions | None = None,
    ) -> DittoRenderMetrics:
        validate_ditto_render_settings(settings)
        if motion is not None:
            motion.validate()
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
                more_kwargs=build_ditto_more_kwargs(settings, motion),
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

    def stream_frames(
        self,
        source_path: Path,
        audio_path: Path,
        output_path: Path,
        *,
        settings: DittoRenderSettings,
        on_frame: Callable[[np.ndarray], None],
        seed: int = 1024,
        should_cancel: Callable[[], bool] | None = None,
        motion: DittoMotionOptions | None = None,
    ) -> DittoStreamMetrics:
        """Run Ditto online mode and publish BGR frames as soon as they exist.

        ``motion`` blends the first/last frames toward the portrait pose, the
        same as the offline render, so a streamed reply starts and ends on
        the neutral frame the idle loop uses.
        """
        validate_ditto_render_settings(settings)
        if motion is not None:
            motion.validate()
        source_path = source_path.resolve()
        audio_path = audio_path.resolve()
        output_path = output_path.resolve()
        if not source_path.is_file():
            raise DittoEngineError(f"Source image not found: {source_path}")
        if not audio_path.is_file():
            raise DittoEngineError(f"Audio file not found: {audio_path}")

        with self._state_lock:
            runtime = self._runtime
            sdk = self._online_sdk
            if not self._loaded or runtime is None:
                raise DittoEngineError("Ditto model is not loaded.")
            if sdk is None or runtime.load_audio is None:
                raise DittoEngineError("Ditto online runtime is unavailable.")
        if not self._render_slot.acquire(blocking=False):
            raise DittoEngineBusyError("Ditto GPU is already rendering.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        with self._state_lock:
            self._busy = True
            self._last_error = None

        started = monotonic()
        frame_count = 0
        stream_started = False
        stream_closed = False

        total_frames = 0
        extra_frames = 0

        def publish(frame: np.ndarray) -> None:
            nonlocal frame_count, extra_frames
            if should_cancel is not None and should_cancel():
                raise DittoEngineError("Ditto online stream was cancelled.")
            if total_frames and frame_count >= total_frames:
                # Frames past the audio length come from the padded final
                # chunk; they are outside the fade-out window and would undo
                # the return to the neutral pose.
                extra_frames += 1
                return
            on_frame(frame)
            frame_count += 1

        try:
            runtime.seed_everything(seed)
            audio = runtime.load_audio(str(audio_path)).astype(np.float32)
            samples_per_frame = 16_000 // 25
            total_frames = max(1, math.ceil(len(audio) / samples_per_frame))
            setup_kwargs: dict[str, Any] = {
                "crop_scale": settings.crop_scale,
                "smo_k_d": settings.smoothing_kernel,
                "sampling_timesteps": settings.sampling_timesteps,
            }
            motion_active = motion is not None and motion.active
            if motion_active:
                setup_kwargs["fade_type"] = motion.fade_type
                setup_kwargs["fade_out_keys"] = tuple(motion.fade_keys)
            sdk.setup(str(source_path), str(output_path), **setup_kwargs)
            stream_started = True
            if motion_active:
                sdk.setup_Nd(
                    N_d=total_frames,
                    fade_in=motion.fade_in_frames or -1,
                    fade_out=motion.fade_out_frames or -1,
                    # A fresh dict per stream: Ditto mutates ctrl_info.
                    ctrl_info={},
                )
            else:
                sdk.setup_Nd(N_d=total_frames)
            sdk.writer = _FrameCallbackWriter(sdk.writer, publish)

            chunksize = (3, 5, 2)
            padded = np.concatenate(
                [np.zeros(chunksize[0] * samples_per_frame, dtype=np.float32), audio]
            )
            split_len = int(sum(chunksize) * 0.04 * 16_000) + 80
            for index in range(0, len(padded), chunksize[1] * samples_per_frame):
                if should_cancel is not None and should_cancel():
                    raise DittoEngineError("Ditto online stream was cancelled.")
                chunk = padded[index : index + split_len]
                if len(chunk) < split_len:
                    chunk = np.pad(chunk, (0, split_len - len(chunk)))
                sdk.run_chunk(chunk, chunksize)
            sdk.close()
            stream_closed = True
            if frame_count <= 0:
                raise DittoEngineError("Ditto online mode produced no frames.")
            if extra_frames:
                print(
                    "[DITTO_SERVICE] online stream trimmed frames past the audio: "
                    f"published={frame_count} trimmed={extra_frames}",
                    flush=True,
                )
            duration_seconds = monotonic() - started
            with self._state_lock:
                self._render_count += 1
                self._last_render_seconds = duration_seconds
                self._last_error = None
            return DittoStreamMetrics(duration_seconds, frame_count)
        except Exception as exc:
            with self._state_lock:
                self._last_error = str(exc)
            if isinstance(exc, DittoEngineError):
                raise
            raise DittoEngineError(f"Ditto online stream failed: {exc}") from exc
        finally:
            if stream_started and not stream_closed:
                try:
                    sdk.close()
                except Exception:
                    pass
            with self._state_lock:
                self._busy = False
            self._render_slot.release()

    def status(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "loaded": self._loaded,
                "busy": self._busy,
                "backend": self.config.normalized_backend,
                "gpu": self._gpu_name,
                "modelLoadSeconds": self._load_seconds,
                "renderCount": self._render_count,
                "streamingAvailable": self._online_sdk is not None,
                "lastRenderSeconds": self._last_render_seconds,
                "lastError": self._last_error,
            }


def _load_runtime(repository_dir: Path) -> DittoRuntime:
    _install_numpy_compat_aliases()
    repository = str(repository_dir)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    os.chdir(repository_dir)
    inference = importlib.import_module("inference")
    pipeline = importlib.import_module("stream_pipeline_offline")
    try:
        online_pipeline = importlib.import_module("stream_pipeline_online")
    except ModuleNotFoundError:
        online_pipeline = None
    torch = importlib.import_module("torch")
    load_audio = None
    if online_pipeline is not None:
        try:
            librosa = importlib.import_module("librosa")
        except ModuleNotFoundError:
            online_pipeline = None
        else:
            def load_audio(path: str) -> np.ndarray:
                audio, _ = librosa.core.load(path, sr=16_000)
                return audio.astype(np.float32)

    return DittoRuntime(
        sdk_factory=pipeline.StreamSDK,
        run=inference.run,
        seed_everything=inference.seed_everything,
        cuda_available=torch.cuda.is_available,
        gpu_name=lambda: torch.cuda.get_device_name(0),
        online_sdk_factory=(
            online_pipeline.StreamSDK if online_pipeline is not None else None
        ),
        load_audio=load_audio,
    )


def _install_numpy_compat_aliases() -> None:
    """Provide NumPy 2 spellings used by Ditto on the pinned NumPy 1 runtime."""
    numpy = importlib.import_module("numpy")
    for alias, target in NUMPY_COMPAT_ALIASES.items():
        if not hasattr(numpy, alias):
            setattr(numpy, alias, getattr(numpy, target))


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    value = str(path)
    if value not in entries:
        os.environ["PATH"] = os.pathsep.join((value, current))
