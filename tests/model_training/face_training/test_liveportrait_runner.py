import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_training.face_training.liveportrait_runner import (
    LivePortraitConfig,
    LivePortraitRunnerError,
    run_liveportrait,
)


class LivePortraitRunnerTests(unittest.TestCase):
    def test_runs_inference_and_returns_generated_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = self._create_repository(root)
            source = root / "source.jpg"
            driving = root / "driving.mp4"
            source.write_bytes(b"source")
            driving.write_bytes(b"driving")
            captured_command = []
            captured_environment = {}

            def fake_runner(command, **kwargs):
                captured_command.extend(command)
                captured_environment.update(kwargs["env"])
                output_dir = Path(command[command.index("-o") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "source--driving.mp4").write_bytes(b"result")
                (output_dir / "source--driving_concat.mp4").write_bytes(
                    b"comparison"
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="completed",
                    stderr="",
                )

            ffmpeg_dir = root / "ffmpeg-bin"
            with mock.patch.dict(
                os.environ,
                {
                    "FFMPEG_BINARY": str(ffmpeg_dir / "ffmpeg"),
                    "FFPROBE_BINARY": str(ffmpeg_dir / "ffprobe"),
                },
            ):
                result = run_liveportrait(
                    source,
                    driving,
                    root / "outputs",
                    config=LivePortraitConfig(
                        repository_dir=repository,
                        python_binary=Path(sys.executable),
                        crop_scale=2.7,
                    ),
                    command_runner=fake_runner,
                )

            self.assertEqual(result.output_path.name, "source--driving.mp4")
            self.assertEqual(
                result.comparison_path.name,
                "source--driving_concat.mp4",
            )
            self.assertIn("--driving_option", captured_command)
            self.assertIn("expression-friendly", captured_command)
            self.assertEqual(
                captured_command[captured_command.index("--scale") + 1],
                "2.7",
            )
            self.assertIn("completed", result.log_path.read_text(encoding="utf-8"))
            self.assertEqual(
                captured_environment["PATH"].split(os.pathsep)[0],
                str(ffmpeg_dir),
            )

    def test_reports_nonzero_inference_exit_and_keeps_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = self._create_repository(root)
            source = root / "source.jpg"
            driving = root / "driving.mp4"
            source.write_bytes(b"source")
            driving.write_bytes(b"driving")
            output_dir = root / "outputs"

            def fake_runner(command, **kwargs):
                return subprocess.CompletedProcess(
                    command,
                    2,
                    stdout="starting",
                    stderr="inference failed",
                )

            with self.assertRaisesRegex(
                LivePortraitRunnerError,
                "exit code 2",
            ):
                run_liveportrait(
                    source,
                    driving,
                    output_dir,
                    config=LivePortraitConfig(
                        repository_dir=repository,
                        python_binary=Path(sys.executable),
                    ),
                    command_runner=fake_runner,
                )

            log = (output_dir / "liveportrait.log").read_text(encoding="utf-8")
            self.assertIn("inference failed", log)

    def test_rejects_repository_without_required_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "liveportrait"
            repository.mkdir()
            (repository / "inference.py").write_text("", encoding="utf-8")
            source = root / "source.jpg"
            driving = root / "driving.mp4"
            source.write_bytes(b"source")
            driving.write_bytes(b"driving")

            with self.assertRaisesRegex(
                LivePortraitRunnerError,
                "weights are missing",
            ):
                run_liveportrait(
                    source,
                    driving,
                    root / "outputs",
                    config=LivePortraitConfig(
                        repository_dir=repository,
                        python_binary=Path(sys.executable),
                    ),
                )

    @staticmethod
    def _create_repository(root: Path) -> Path:
        repository = root / "liveportrait"
        (repository / "pretrained_weights/liveportrait/base_models").mkdir(
            parents=True
        )
        (
            repository
            / "pretrained_weights/liveportrait/retargeting_models"
        ).mkdir(parents=True)
        (repository / "inference.py").write_text("", encoding="utf-8")
        required_weights = (
            "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
            "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
            "pretrained_weights/liveportrait/base_models/warping_module.pth",
            "pretrained_weights/liveportrait/base_models/spade_generator.pth",
            "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
            "pretrained_weights/liveportrait/landmark.onnx",
        )
        for relative_path in required_weights:
            path = repository / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weight")
        return repository


if __name__ == "__main__":
    unittest.main()
