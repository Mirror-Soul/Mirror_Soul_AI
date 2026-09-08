import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from model_training.face_training.musetalk_runner import (
    MuseTalkConfig,
    run_musetalk_preview,
)
from model_training.face_training.natural_motion import (
    NaturalMotionConfig,
    prepare_natural_motion_clip,
)


load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a natural-motion MuseTalk preview from an existing face "
            "preprocess manifest and member audio."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--bbox-shift",
        type=int,
        default=int(os.getenv("FACE_TRAINING_MUSETALK_BBOX_SHIFT", "0")),
    )
    args = parser.parse_args()

    ffmpeg_binary = os.getenv("FFMPEG_BINARY", "ffmpeg")
    ffprobe_binary = os.getenv("FFPROBE_BINARY", "ffprobe")
    source_result = prepare_natural_motion_clip(
        args.manifest,
        args.audio,
        args.output_dir / "natural-motion-source.mp4",
        config=NaturalMotionConfig(
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
        ),
    )
    print(
        "[NATURAL_MOTION] source clip prepared: "
        f"start={source_result.start_seconds:.2f}s "
        f"duration={source_result.duration_seconds:.2f}s "
        f"front_ratio={source_result.front_frame_ratio:.2f} "
        f"confidence={source_result.confidence}",
        flush=True,
    )

    musetalk_result = run_musetalk_preview(
        source_result.clip_path,
        args.audio,
        args.output_dir / "musetalk",
        config=MuseTalkConfig(
            repository_dir=Path(
                os.getenv(
                    "FACE_TRAINING_MUSETALK_REPO_DIR",
                    "/shareHost/C084003-musetalk/MuseTalk",
                )
            ),
            python_binary=Path(
                os.getenv(
                    "FACE_TRAINING_MUSETALK_PYTHON",
                    "/shareHost/C084003-musetalk/conda-env/bin/python",
                )
            ),
            timeout_seconds=int(
                os.getenv("FACE_TRAINING_MUSETALK_TIMEOUT_SECONDS", "900")
            ),
            bbox_shift=args.bbox_shift,
            ffmpeg_dir=Path(
                os.getenv("FACE_TRAINING_MUSETALK_FFMPEG_DIR", "/opt/conda/bin")
            ),
        ),
    )
    print(
        "[NATURAL_MOTION] MuseTalk completed: "
        f"output={musetalk_result.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
