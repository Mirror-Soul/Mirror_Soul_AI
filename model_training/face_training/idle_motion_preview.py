import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from model_training.face_training.natural_motion import (
    NaturalMotionConfig,
    prepare_natural_idle_clip,
)


load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a long, silent idle-motion preview for a face profile."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--segment-duration", type=float, default=2.5)
    args = parser.parse_args()

    result = prepare_natural_idle_clip(
        args.manifest,
        args.output,
        output_duration_seconds=args.duration,
        segment_duration_seconds=args.segment_duration,
        config=NaturalMotionConfig(
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
            ffprobe_binary=os.getenv("FFPROBE_BINARY", "ffprobe"),
            output_fps=int(os.getenv("FACE_TRAINING_NATURAL_MOTION_FPS", "25")),
            crf=int(os.getenv("FACE_TRAINING_NATURAL_MOTION_CRF", "18")),
            minimum_front_ratio=float(
                os.getenv("FACE_TRAINING_NATURAL_MOTION_MIN_FRONT_RATIO", "0.60")
            ),
        ),
    )
    print(
        "[IDLE_MOTION] completed: "
        f"output={result.output_path} "
        f"duration={result.output_duration_seconds:.2f}s "
        f"segment={result.segment_duration_seconds:.2f}s "
        f"front_ratio={result.front_frame_ratio:.2f} "
        f"confidence={result.confidence}",
        flush=True,
    )


if __name__ == "__main__":
    main()
