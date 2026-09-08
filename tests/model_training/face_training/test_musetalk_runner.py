import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_training.face_training.musetalk_runner import (
    MuseTalkConfig,
    MuseTalkRunnerError,
    run_musetalk_preview,
)


class MuseTalkRunnerTests(unittest.TestCase):
    def _paths(self, directory: str):
        root = Path(directory)
        repository = root / "MuseTalk"
        (repository / "models" / "musetalkV15").mkdir(parents=True)
        (repository / "models" / "musetalkV15" / "unet.pth").write_bytes(b"m")
        (repository / "models" / "musetalkV15" / "musetalk.json").write_text(
            "{}", encoding="utf-8"
        )
        python = root / "conda-env" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        source = root / "source.jpg"
        source.write_bytes(b"image")
        audio = root / "member.mp3"
        audio.write_bytes(b"audio")
        output = root / "output"
        return repository, python, source, audio, output

    def test_runs_musetalk_with_structured_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, python, source, audio, output = self._paths(directory)
            expected = output / "v15" / "source_member.mp4"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"video")

            with patch(
                "model_training.face_training.musetalk_runner.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout="completed",
                    stderr="",
                ),
            ) as run:
                result = run_musetalk_preview(
                    source,
                    audio,
                    output,
                    config=MuseTalkConfig(
                        repository_dir=repository,
                        python_binary=python,
                        bbox_shift=3,
                    ),
                )

            config = json.loads(result.inference_config_path.read_text("utf-8"))
            self.assertEqual(config["task_0"]["bbox_shift"], 3)
            self.assertEqual(result.output_path, expected.resolve())
            self.assertEqual(run.call_args.kwargs["cwd"], repository)

    def test_fails_when_process_does_not_create_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, python, source, audio, output = self._paths(directory)
            with patch(
                "model_training.face_training.musetalk_runner.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="failed",
                ),
            ):
                with self.assertRaisesRegex(
                    MuseTalkRunnerError,
                    "did not produce",
                ):
                    run_musetalk_preview(
                        source,
                        audio,
                        output,
                        config=MuseTalkConfig(
                            repository_dir=repository,
                            python_binary=python,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
