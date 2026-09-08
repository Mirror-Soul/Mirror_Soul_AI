import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MuseTalkRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class MuseTalkConfig:
    repository_dir: Path
    python_binary: Path
    timeout_seconds: int = 900
    bbox_shift: int = 0
    version: str = "v15"
    ffmpeg_dir: Path = Path("/opt/conda/bin")


@dataclass(frozen=True)
class MuseTalkResult:
    source_path: Path
    audio_path: Path
    output_path: Path
    log_path: Path
    inference_config_path: Path
    bbox_shift: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sourcePath": str(self.source_path),
            "audioPath": str(self.audio_path),
            "outputPath": str(self.output_path),
            "logPath": str(self.log_path),
            "inferenceConfigPath": str(self.inference_config_path),
            "bboxShift": self.bbox_shift,
        }


def run_musetalk_preview(
    source_path: Path,
    audio_path: Path,
    output_dir: Path,
    *,
    config: MuseTalkConfig,
) -> MuseTalkResult:
    source_path = source_path.resolve()
    audio_path = audio_path.resolve()
    output_dir = output_dir.resolve()
    _validate_inputs(source_path, audio_path, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_config_path = output_dir / "inference-config.json"
    inference_config_path.write_text(
        json.dumps(
            {
                "task_0": {
                    "video_path": str(source_path),
                    "audio_path": str(audio_path),
                    "bbox_shift": config.bbox_shift,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = (
        str(config.python_binary),
        "-m",
        "scripts.inference",
        "--inference_config",
        str(inference_config_path),
        "--result_dir",
        str(output_dir),
        "--unet_model_path",
        "./models/musetalkV15/unet.pth",
        "--unet_config",
        "./models/musetalkV15/musetalk.json",
        "--version",
        config.version,
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = os.pathsep.join(
        (
            str(config.python_binary.parent),
            str(config.ffmpeg_dir),
            environment.get("PATH", ""),
        )
    )

    try:
        completed = subprocess.run(
            command,
            cwd=config.repository_dir,
            env=environment,
            text=True,
            capture_output=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MuseTalkRunnerError(f"MuseTalk execution failed: {exc}") from exc

    log_path = output_dir / "musetalk.log"
    log_path.write_text(
        "[stdout]\n"
        + completed.stdout
        + "\n\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    output_path = _find_output(
        output_dir,
        source_stem=source_path.stem,
        audio_stem=audio_path.stem,
        version=config.version,
    )
    if output_path is None:
        raise MuseTalkRunnerError(
            "MuseTalk did not produce an output video. "
            f"exit_code={completed.returncode} log={log_path}"
        )

    return MuseTalkResult(
        source_path=source_path,
        audio_path=audio_path,
        output_path=output_path,
        log_path=log_path,
        inference_config_path=inference_config_path,
        bbox_shift=config.bbox_shift,
    )


def _validate_inputs(
    source_path: Path,
    audio_path: Path,
    config: MuseTalkConfig,
) -> None:
    for path, label in (
        (source_path, "source image/video"),
        (audio_path, "audio"),
        (config.repository_dir, "MuseTalk repository"),
        (config.python_binary, "MuseTalk Python"),
    ):
        if not path.exists():
            raise MuseTalkRunnerError(f"{label} not found: {path}")

    for relative_path in (
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
    ):
        model_path = config.repository_dir / relative_path
        if not model_path.is_file():
            raise MuseTalkRunnerError(f"MuseTalk model not found: {model_path}")


def _find_output(
    output_dir: Path,
    *,
    source_stem: str,
    audio_stem: str,
    version: str,
) -> Path | None:
    expected = output_dir / version / f"{source_stem}_{audio_stem}.mp4"
    if expected.is_file() and expected.stat().st_size > 0:
        return expected

    candidates = sorted(
        path
        for path in output_dir.rglob("*.mp4")
        if not path.name.startswith("temp_") and path.stat().st_size > 0
    )
    return candidates[-1] if candidates else None
