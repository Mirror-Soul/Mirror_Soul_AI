from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


class LivePortraitRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LivePortraitConfig:
    repository_dir: Path
    python_binary: Path
    timeout_seconds: int = 900
    driving_option: str = "expression-friendly"
    driving_multiplier: float = 1.0
    source_max_dim: int = 1280
    source_division: int = 2


@dataclass(frozen=True)
class LivePortraitResult:
    source_path: Path
    driving_path: Path
    output_path: Path
    comparison_path: Path | None
    log_path: Path
    duration_seconds: float
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourcePath": str(self.source_path),
            "drivingPath": str(self.driving_path),
            "outputPath": str(self.output_path),
            "comparisonPath": (
                str(self.comparison_path)
                if self.comparison_path is not None
                else None
            ),
            "logPath": str(self.log_path),
            "durationSeconds": round(self.duration_seconds, 3),
            "command": list(self.command),
        }


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def run_liveportrait(
    source_path: Path,
    driving_path: Path,
    output_dir: Path,
    *,
    config: LivePortraitConfig,
    command_runner: CommandRunner = subprocess.run,
    extra_arguments: Sequence[str] = (),
) -> LivePortraitResult:
    source = source_path.resolve()
    driving = driving_path.resolve()
    repository = config.repository_dir.resolve()
    python_binary = config.python_binary.resolve()
    output = output_dir.resolve()

    _validate_inputs(
        source=source,
        driving=driving,
        repository=repository,
        python_binary=python_binary,
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "liveportrait.log"

    command = (
        str(python_binary),
        str(repository / "inference.py"),
        "-s",
        str(source),
        "-d",
        str(driving),
        "-o",
        str(output),
        "--driving_option",
        config.driving_option,
        "--driving_multiplier",
        str(config.driving_multiplier),
        "--source_max_dim",
        str(config.source_max_dim),
        "--source_division",
        str(config.source_division),
        *tuple(extra_arguments),
    )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started_at = time.monotonic()
    try:
        completed = command_runner(
            list(command),
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _write_log(log_path, exc.stdout or "", exc.stderr or "")
        raise LivePortraitRunnerError(
            f"LivePortrait timed out after {config.timeout_seconds} seconds."
        ) from exc
    except OSError as exc:
        raise LivePortraitRunnerError(f"Unable to start LivePortrait: {exc}") from exc

    duration_seconds = time.monotonic() - started_at
    _write_log(log_path, completed.stdout or "", completed.stderr or "")
    if completed.returncode != 0:
        raise LivePortraitRunnerError(
            "LivePortrait failed with exit code "
            f"{completed.returncode}. See log: {log_path}"
        )

    generated_videos = sorted(output.glob("*.mp4"))
    comparison_path = next(
        (path for path in generated_videos if path.stem.endswith("_concat")),
        None,
    )
    output_path = next(
        (path for path in generated_videos if not path.stem.endswith("_concat")),
        None,
    )
    if output_path is None:
        raise LivePortraitRunnerError(
            f"LivePortrait completed without a generated video. See log: {log_path}"
        )

    return LivePortraitResult(
        source_path=source,
        driving_path=driving,
        output_path=output_path,
        comparison_path=comparison_path,
        log_path=log_path,
        duration_seconds=duration_seconds,
        command=command,
    )


def _validate_inputs(
    *,
    source: Path,
    driving: Path,
    repository: Path,
    python_binary: Path,
) -> None:
    if not source.is_file():
        raise LivePortraitRunnerError(f"Source portrait is missing: {source}")
    if not driving.is_file():
        raise LivePortraitRunnerError(f"Driving video is missing: {driving}")
    if not (repository / "inference.py").is_file():
        raise LivePortraitRunnerError(
            f"LivePortrait repository is invalid: {repository}"
        )
    if not python_binary.is_file():
        raise LivePortraitRunnerError(
            f"LivePortrait Python executable is missing: {python_binary}"
        )

    required_weights = (
        repository
        / "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        repository
        / "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        repository
        / "pretrained_weights/liveportrait/base_models/warping_module.pth",
        repository
        / "pretrained_weights/liveportrait/base_models/spade_generator.pth",
        repository
        / "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        repository / "pretrained_weights/liveportrait/landmark.onnx",
    )
    missing_weights = [str(path) for path in required_weights if not path.is_file()]
    if missing_weights:
        raise LivePortraitRunnerError(
            "LivePortrait weights are missing: " + ", ".join(missing_weights)
        )


def _write_log(log_path: Path, stdout: str, stderr: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "[stdout]\n" + stdout + "\n\n[stderr]\n" + stderr,
        encoding="utf-8",
    )
