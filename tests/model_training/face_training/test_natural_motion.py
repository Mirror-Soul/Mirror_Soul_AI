import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_training.face_training.natural_motion import (
    NaturalMotionConfig,
    prepare_natural_motion_clip,
    select_natural_motion_window,
)


class NaturalMotionTests(unittest.TestCase):
    def test_prefers_contiguous_front_facing_window(self) -> None:
        profile = _frame("left_profile", 90.0)
        front = _frame("front", 65.0)
        videos = [
            {
                "localPath": "/tmp/member.mov",
                "metadata": {"duration_seconds": 5.0},
                "frameSelection": {
                    "frames": [profile, profile, front, front, front]
                },
            }
        ]

        result = select_natural_motion_window(
            videos,
            target_duration_seconds=2.0,
        )

        self.assertEqual(result.start_seconds, 2.0)
        self.assertEqual(result.front_frame_ratio, 1.0)
        self.assertEqual(result.confidence, "high")

    def test_penalizes_windows_without_reliable_face_detection(self) -> None:
        missing = _frame(
            "unknown",
            80.0,
            reasons=["face_not_detected"],
        )
        front = _frame("front", 55.0, reasons=["too_blurry"])
        videos = [
            {
                "localPath": "/tmp/member.mov",
                "metadata": {"duration_seconds": 4.0},
                "frameSelection": {"frames": [missing, missing, front, front]},
            }
        ]

        result = select_natural_motion_window(
            videos,
            target_duration_seconds=2.0,
        )

        self.assertEqual(result.start_seconds, 2.0)
        self.assertEqual(result.front_frame_ratio, 1.0)
        self.assertEqual(result.confidence, "medium")

    def test_prepares_video_clip_matching_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "member.mov"
            source.write_bytes(b"video")
            audio = root / "member.mp3"
            audio.write_bytes(b"audio")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "localPath": str(source),
                                "metadata": {"duration_seconds": 5.0},
                                "frameSelection": {
                                    "frames": [
                                        _frame("front", 70.0) for _ in range(5)
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "natural.mp4"

            def run(command, **kwargs):
                if command[0] == "probe":
                    return SimpleNamespace(
                        stdout='{"format":{"duration":"3.096"}}'
                    )
                output.write_bytes(b"clip")
                return SimpleNamespace(stdout="", stderr="")

            with patch(
                "model_training.face_training.natural_motion.subprocess.run",
                side_effect=run,
            ) as subprocess_run:
                result = prepare_natural_motion_clip(
                    manifest,
                    audio,
                    output,
                    config=NaturalMotionConfig(
                        ffmpeg_binary="encode",
                        ffprobe_binary="probe",
                    ),
                )

            self.assertEqual(result.duration_seconds, 3.096)
            self.assertEqual(result.clip_path, output.resolve())
            ffmpeg_command = subprocess_run.call_args_list[1].args[0]
            self.assertIn("3.096", ffmpeg_command)
            self.assertIn("fps=25", ffmpeg_command)


def _frame(
    view: str,
    quality: float,
    *,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "view": view,
        "qualityScore": quality,
        "accepted": not reasons,
        "rejectionReasons": reasons or [],
    }


if __name__ == "__main__":
    unittest.main()
