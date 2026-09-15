import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_training.face_training.ditto_runner import (
    DittoConfig,
    DittoRenderSettings,
    DittoRunnerError,
    load_ditto_render_settings,
    run_ditto_preview,
)


class DittoRunnerTests(unittest.TestCase):
    def _paths(self, directory: str):
        root = Path(directory)
        repository = root / "ditto-talkinghead"
        data_root = repository / "checkpoints" / "ditto_pytorch"
        data_root.mkdir(parents=True)
        config_path = (
            repository
            / "checkpoints"
            / "ditto_cfg"
            / "v0.4_hubert_cfg_pytorch.pkl"
        )
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes(b"config")
        python = root / "conda-env" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        source = root / "portrait.jpg"
        source.write_bytes(b"image")
        audio = root / "speech.wav"
        audio.write_bytes(b"audio")
        output = root / "results" / "preview.mp4"
        return repository, python, source, audio, output

    def test_runs_ditto_with_selected_smooth_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, python, source, audio, output = self._paths(directory)

            def complete(command, **kwargs):
                rendered_output = Path(command[command.index("--output-path") + 1])
                rendered_output.parent.mkdir(parents=True, exist_ok=True)
                rendered_output.write_bytes(b"video")
                return SimpleNamespace(returncode=0, stdout="completed", stderr="")

            with patch(
                "model_training.face_training.ditto_runner.subprocess.run",
                side_effect=complete,
            ) as run:
                result = run_ditto_preview(
                    source,
                    audio,
                    output,
                    config=DittoConfig(
                        repository_dir=repository,
                        python_binary=python,
                    ),
                    settings=DittoRenderSettings(
                        crop_scale=2.3,
                        smoothing_kernel=5,
                        sampling_timesteps=50,
                    ),
                )

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--crop-scale") + 1], "2.3")
            self.assertEqual(command[command.index("--smo-k-d") + 1], "5")
            self.assertEqual(
                command[command.index("--sampling-timesteps") + 1], "50"
            )
            self.assertEqual(run.call_args.kwargs["cwd"], repository.resolve())
            self.assertEqual(result.output_path, output.resolve())
            self.assertTrue(result.log_path.is_file())

    def test_loads_settings_from_face_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "face-profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "engine": {
                            "name": "ditto",
                            "renderSettings": {
                                "cropScale": 2.3,
                                "smoothingKernel": 5,
                                "samplingTimesteps": 50,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = load_ditto_render_settings(profile)

            self.assertEqual(settings, DittoRenderSettings(2.3, 5, 50))

    def test_rejects_non_ditto_face_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "face-profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "engine": {
                            "name": "liveportrait",
                            "renderSettings": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DittoRunnerError, "must be ditto"):
                load_ditto_render_settings(profile)

    def test_rejects_even_smoothing_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, python, source, audio, output = self._paths(directory)
            with self.assertRaisesRegex(DittoRunnerError, "odd integer"):
                run_ditto_preview(
                    source,
                    audio,
                    output,
                    config=DittoConfig(
                        repository_dir=repository,
                        python_binary=python,
                    ),
                    settings=DittoRenderSettings(smoothing_kernel=4),
                )

    def test_reports_process_failure_with_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, python, source, audio, output = self._paths(directory)
            output.parent.mkdir(parents=True)
            output.write_bytes(b"existing-video")
            with patch(
                "model_training.face_training.ditto_runner.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="render failed",
                ),
            ):
                with self.assertRaisesRegex(
                    DittoRunnerError,
                    "exited unsuccessfully",
                ):
                    run_ditto_preview(
                        source,
                        audio,
                        output,
                        config=DittoConfig(
                            repository_dir=repository,
                            python_binary=python,
                        ),
                    )
            self.assertTrue(output.with_suffix(".ditto.log").is_file())
            self.assertEqual(output.read_bytes(), b"existing-video")


if __name__ == "__main__":
    unittest.main()
