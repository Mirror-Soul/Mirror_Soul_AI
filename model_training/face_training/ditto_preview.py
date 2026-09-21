from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from model_training.face_training.ditto_runner import (
    DittoConfig,
    DittoRenderSettings,
    load_ditto_render_settings,
    run_ditto_preview,
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Render a Ditto preview without SQS or backend dependencies."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--crop-scale", type=float)
    parser.add_argument("--smo-k-d", type=int)
    parser.add_argument("--smo-k-s", type=int)
    parser.add_argument("--blink-open-frames", type=int)
    parser.add_argument(
        "--drive-eye",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--blink-strength", type=float)
    parser.add_argument("--sampling-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    settings = (
        load_ditto_render_settings(args.profile)
        if args.profile is not None
        else DittoRenderSettings(
            crop_scale=_env_float("FACE_TRAINING_DITTO_CROP_SCALE", 2.3),
            smoothing_kernel=_env_int("FACE_TRAINING_DITTO_SMO_K_D", 5),
            sampling_timesteps=_env_int(
                "FACE_TRAINING_DITTO_SAMPLING_TIMESTEPS", 50
            ),
        )
    )
    if args.crop_scale is not None:
        settings = replace(settings, crop_scale=args.crop_scale)
    if args.smo_k_d is not None:
        settings = replace(settings, smoothing_kernel=args.smo_k_d)
    if args.sampling_timesteps is not None:
        settings = replace(
            settings,
            sampling_timesteps=args.sampling_timesteps,
        )

    repository_dir = Path(
        os.getenv(
            "FACE_TRAINING_DITTO_REPO_DIR",
            "/shareHost/C084003-ditto/ditto-talkinghead",
        )
    )
    result = run_ditto_preview(
        args.source,
        args.audio,
        args.output,
        config=DittoConfig(
            repository_dir=repository_dir,
            python_binary=Path(
                os.getenv(
                    "FACE_TRAINING_DITTO_PYTHON",
                    "/shareHost/C084003-ditto/conda-env/bin/python",
                )
            ),
            data_root=_env_path("FACE_TRAINING_DITTO_DATA_ROOT"),
            config_path=_env_path("FACE_TRAINING_DITTO_CONFIG_PATH"),
            timeout_seconds=_env_int("FACE_TRAINING_DITTO_TIMEOUT_SECONDS", 1800),
            seed=(
                args.seed
                if args.seed is not None
                else _env_int("FACE_TRAINING_DITTO_SEED", 1024)
            ),
            ffmpeg_dir=Path(
                os.getenv("FACE_TRAINING_DITTO_FFMPEG_DIR", "/opt/conda/bin")
            ),
        ),
        settings=settings,
        source_smoothing_kernel=(
            args.smo_k_s
            if args.smo_k_s is not None
            else _env_int("FACE_TRAINING_DITTO_SMO_K_S", 13)
        ),
        blink_open_frames=(
            args.blink_open_frames
            if args.blink_open_frames is not None
            else _env_int("FACE_TRAINING_DITTO_BLINK_OPEN_FRAMES", 0)
        ),
        drive_eye=args.drive_eye,
        blink_strength=(
            args.blink_strength
            if args.blink_strength is not None
            else _env_float("FACE_TRAINING_DITTO_BLINK_STRENGTH", 1.0)
        ),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


if __name__ == "__main__":
    main()
