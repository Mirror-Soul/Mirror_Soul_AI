from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Ditto offline render with explicit settings."
    )
    parser.add_argument("--repository-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--crop-scale", type=float, required=True)
    parser.add_argument("--smo-k-d", type=int, required=True)
    parser.add_argument("--smo-k-s", type=int, default=13)
    parser.add_argument("--blink-open-frames", type=int, default=0)
    parser.add_argument(
        "--drive-eye",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--blink-strength", type=float, default=1.0)
    parser.add_argument("--sampling-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    repository_dir = Path(args.repository_dir).resolve()
    sys.path.insert(0, str(repository_dir))
    os.chdir(repository_dir)

    from inference import run, seed_everything
    from stream_pipeline_offline import StreamSDK

    seed_everything(args.seed)
    sdk = StreamSDK(args.config_path, args.data_root)
    if args.blink_strength != 1.0:
        delta_eye_arr = sdk.default_kwargs.get("delta_eye_arr")
        if delta_eye_arr is None:
            raise RuntimeError("Ditto config does not contain delta_eye_arr.")
        sdk.default_kwargs["delta_eye_arr"] = (
            delta_eye_arr * args.blink_strength
        )
    run(
        sdk,
        args.audio_path,
        args.source_path,
        args.output_path,
        more_kwargs={
            "setup_kwargs": {
                "crop_scale": args.crop_scale,
                "smo_k_d": args.smo_k_d,
                "smo_k_s": args.smo_k_s,
                "delta_eye_open_n": args.blink_open_frames,
                "drive_eye": args.drive_eye,
                "sampling_timesteps": args.sampling_timesteps,
            }
        },
    )


if __name__ == "__main__":
    main()
