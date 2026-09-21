from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class DittoRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DittoRenderSettings:
    crop_scale: float = 2.3
    smoothing_kernel: int = 5
    sampling_timesteps: int = 50

    def to_dict(self) -> dict[str, object]:
        return {
            "cropScale": self.crop_scale,
            "smoothingKernel": self.smoothing_kernel,
            "samplingTimesteps": self.sampling_timesteps,
        }


@dataclass(frozen=True)
class DittoConfig:
    repository_dir: Path
    python_binary: Path
    data_root: Path | None = None
    config_path: Path | None = None
    timeout_seconds: int = 1800
    seed: int = 1024
    ffmpeg_dir: Path = Path("/opt/conda/bin")


@dataclass(frozen=True)
class DittoResult:
    source_path: Path
    audio_path: Path
    output_path: Path
    log_path: Path
    settings: DittoRenderSettings
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sourcePath": str(self.source_path),
            "audioPath": str(self.audio_path),
            "outputPath": str(self.output_path),
            "logPath": str(self.log_path),
            "renderSettings": self.settings.to_dict(),
            "seed": self.seed,
        }


def load_ditto_render_settings(profile_path: Path) -> DittoRenderSettings:
    profile_path = profile_path.resolve()
    if not profile_path.is_file():
        raise DittoRunnerError(f"Face profile not found: {profile_path}")

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DittoRunnerError(
            f"Invalid face profile JSON: {profile_path}"
        ) from exc

    engine = profile.get("engine") or {}
    if str(engine.get("name", "")).strip().lower() != "ditto":
        raise DittoRunnerError("Face profile engine must be ditto.")

    values = engine.get("renderSettings") or {}
    try:
        settings = DittoRenderSettings(
            crop_scale=float(values["cropScale"]),
            smoothing_kernel=int(values["smoothingKernel"]),
            sampling_timesteps=int(values["samplingTimesteps"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DittoRunnerError(
            "Face profile is missing valid Ditto render settings."
        ) from exc

    validate_ditto_render_settings(settings)
    return settings


def run_ditto_preview(
    source_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    config: DittoConfig,
    settings: DittoRenderSettings = DittoRenderSettings(),
    source_smoothing_kernel: int = 13,
    blink_open_frames: int = 0,
    drive_eye: bool | None = None,
    blink_strength: float = 1.0,
) -> DittoResult:
    source_path = source_path.resolve()
    audio_path = audio_path.resolve()
    output_path = output_path.resolve()
    repository_dir = config.repository_dir.resolve()
    python_binary = config.python_binary.resolve()
    data_root = _resolve_from_repository(
        repository_dir,
        config.data_root,
        "checkpoints/ditto_pytorch",
    )
    config_path = _resolve_from_repository(
        repository_dir,
        config.config_path,
        "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    )
    entry_path = Path(__file__).with_name("ditto_inference_entry.py").resolve()

    _validate_inputs(
        source_path=source_path,
        audio_path=audio_path,
        output_path=output_path,
        repository_dir=repository_dir,
        python_binary=python_binary,
        data_root=data_root,
        config_path=config_path,
        entry_path=entry_path,
        config=config,
        settings=settings,
        source_smoothing_kernel=source_smoothing_kernel,
        blink_open_frames=blink_open_frames,
        blink_strength=blink_strength,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp{output_path.suffix}"
    )

    command = [
        str(python_binary),
        str(entry_path),
        "--repository-dir",
        str(repository_dir),
        "--data-root",
        str(data_root),
        "--config-path",
        str(config_path),
        "--source-path",
        str(source_path),
        "--audio-path",
        str(audio_path),
        "--output-path",
        str(temporary_output_path),
        "--crop-scale",
        str(settings.crop_scale),
        "--smo-k-d",
        str(settings.smoothing_kernel),
        "--smo-k-s",
        str(source_smoothing_kernel),
        "--blink-open-frames",
        str(blink_open_frames),
        "--blink-strength",
        str(blink_strength),
        "--sampling-timesteps",
        str(settings.sampling_timesteps),
        "--seed",
        str(config.seed),
    ]
    if drive_eye is not None:
        command.append("--drive-eye" if drive_eye else "--no-drive-eye")
    command = tuple(command)
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = os.pathsep.join(
        (
            str(python_binary.parent),
            str(config.ffmpeg_dir),
            environment.get("PATH", ""),
        )
    )

    try:
        completed = subprocess.run(
            command,
            cwd=repository_dir,
            env=environment,
            text=True,
            capture_output=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DittoRunnerError(f"Ditto execution failed: {exc}") from exc

    log_path = output_path.with_suffix(".ditto.log")
    log_path.write_text(
        "[stdout]\n"
        + completed.stdout
        + "\n\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        temporary_output_path.unlink(missing_ok=True)
        raise DittoRunnerError(
            "Ditto exited unsuccessfully. "
            f"exit_code={completed.returncode} log={log_path}"
        )
    if (
        not temporary_output_path.is_file()
        or temporary_output_path.stat().st_size <= 0
    ):
        temporary_output_path.unlink(missing_ok=True)
        raise DittoRunnerError(
            f"Ditto did not produce an output video. log={log_path}"
        )
    try:
        temporary_output_path.replace(output_path)
    except OSError as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DittoRunnerError(
            f"Ditto output could not be finalized: {output_path}"
        ) from exc

    return DittoResult(
        source_path=source_path,
        audio_path=audio_path,
        output_path=output_path,
        log_path=log_path,
        settings=settings,
        seed=config.seed,
    )


def _resolve_from_repository(
    repository_dir: Path,
    configured_path: Path | None,
    default_relative_path: str,
) -> Path:
    path = configured_path or Path(default_relative_path)
    return path.resolve() if path.is_absolute() else (repository_dir / path).resolve()


def _validate_inputs(
    *,
    source_path: Path,
    audio_path: Path,
    output_path: Path,
    repository_dir: Path,
    python_binary: Path,
    data_root: Path,
    config_path: Path,
    entry_path: Path,
    config: DittoConfig,
    settings: DittoRenderSettings,
    source_smoothing_kernel: int,
    blink_open_frames: int,
    blink_strength: float,
) -> None:
    validate_ditto_render_settings(settings)
    if source_smoothing_kernel <= 0 or source_smoothing_kernel % 2 == 0:
        raise DittoRunnerError(
            "Ditto source smoothing kernel must be a positive odd integer."
        )
    if blink_open_frames < 0:
        raise DittoRunnerError(
            "Ditto blink open frames must be zero or a positive integer."
        )
    if not 0 < blink_strength <= 1:
        raise DittoRunnerError(
            "Ditto blink strength must be greater than zero and at most one."
        )
    if config.timeout_seconds <= 0:
        raise DittoRunnerError("Ditto timeout must be positive.")
    if output_path.suffix.lower() != ".mp4":
        raise DittoRunnerError("Ditto output path must use the .mp4 extension.")

    required_files = (
        (source_path, "source image/video"),
        (audio_path, "audio"),
        (python_binary, "Ditto Python"),
        (config_path, "Ditto config"),
        (entry_path, "Ditto inference entrypoint"),
    )
    for path, label in required_files:
        if not path.is_file():
            raise DittoRunnerError(f"{label} not found: {path}")
    for path, label in (
        (repository_dir, "Ditto repository"),
        (data_root, "Ditto checkpoint directory"),
    ):
        if not path.is_dir():
            raise DittoRunnerError(f"{label} not found: {path}")


def validate_ditto_render_settings(settings: DittoRenderSettings) -> None:
    if settings.crop_scale <= 0:
        raise DittoRunnerError("Ditto crop scale must be positive.")
    if settings.smoothing_kernel <= 0 or settings.smoothing_kernel % 2 == 0:
        raise DittoRunnerError(
            "Ditto smoothing kernel must be a positive odd integer."
        )
    if settings.sampling_timesteps <= 0:
        raise DittoRunnerError("Ditto sampling timesteps must be positive.")
