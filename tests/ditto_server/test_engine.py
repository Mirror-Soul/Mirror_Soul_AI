import tempfile
import unittest
from pathlib import Path

from ditto_server.config import DittoServiceConfig, DittoServiceConfigError
from ditto_server.engine import (
    DittoEngine,
    DittoEngineBusyError,
    DittoEngineError,
    DittoRuntime,
)
from model_training.face_training.ditto_runner import DittoRenderSettings


class DittoEngineTests(unittest.TestCase):
    def _config(self, directory: str) -> DittoServiceConfig:
        root = Path(directory)
        repository = root / "ditto"
        (repository / "checkpoints" / "ditto_pytorch").mkdir(parents=True)
        model_config = (
            repository
            / "checkpoints"
            / "ditto_cfg"
            / "v0.4_hubert_cfg_pytorch.pkl"
        )
        model_config.parent.mkdir(parents=True)
        model_config.write_bytes(b"config")
        ffmpeg_dir = root / "ffmpeg"
        ffmpeg_dir.mkdir()
        return DittoServiceConfig(
            api_key="test-key",
            repository_dir=repository,
            ffmpeg_dir=ffmpeg_dir,
        )

    def _inputs(self, directory: str):
        root = Path(directory)
        source = root / "portrait.jpg"
        source.write_bytes(b"image")
        audio = root / "speech.wav"
        audio.write_bytes(b"audio")
        output = root / "result.mp4"
        return source, audio, output

    def test_loads_model_once_and_reuses_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            source, audio, first_output = self._inputs(directory)
            second_output = Path(directory) / "second.mp4"
            sdk = object()
            sdk_calls = []
            render_calls = []

            def sdk_factory(config_path, data_root):
                sdk_calls.append((config_path, data_root))
                return sdk

            def run(used_sdk, audio_path, source_path, output_path, **kwargs):
                render_calls.append((used_sdk, kwargs))
                Path(output_path).write_bytes(b"video")

            runtime = DittoRuntime(
                sdk_factory=sdk_factory,
                run=run,
                seed_everything=lambda seed: None,
                cuda_available=lambda: True,
                gpu_name=lambda: "Test GPU",
            )
            engine = DittoEngine(
                config,
                runtime_loader=lambda repository: runtime,
            )

            engine.load()
            engine.load()
            engine.render(
                source,
                audio,
                first_output,
                settings=DittoRenderSettings(),
            )
            engine.render(
                source,
                audio,
                second_output,
                settings=DittoRenderSettings(),
            )

            self.assertEqual(len(sdk_calls), 1)
            self.assertEqual([call[0] for call in render_calls], [sdk, sdk])
            self.assertEqual(engine.status()["renderCount"], 2)
            self.assertEqual(engine.status()["gpu"], "Test GPU")

    def test_rejects_render_while_gpu_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            source, audio, output = self._inputs(directory)
            runtime = DittoRuntime(
                sdk_factory=lambda config_path, data_root: object(),
                run=lambda *args, **kwargs: None,
                seed_everything=lambda seed: None,
                cuda_available=lambda: True,
                gpu_name=lambda: "Test GPU",
            )
            engine = DittoEngine(
                config,
                runtime_loader=lambda repository: runtime,
            )
            engine.load()
            engine._render_slot.acquire()
            try:
                with self.assertRaises(DittoEngineBusyError):
                    engine.render(
                        source,
                        audio,
                        output,
                        settings=DittoRenderSettings(),
                    )
            finally:
                engine._render_slot.release()

    def test_fails_when_runtime_does_not_create_final_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            source, audio, output = self._inputs(directory)
            runtime = DittoRuntime(
                sdk_factory=lambda config_path, data_root: object(),
                run=lambda *args, **kwargs: None,
                seed_everything=lambda seed: None,
                cuda_available=lambda: True,
                gpu_name=lambda: "Test GPU",
            )
            engine = DittoEngine(
                config,
                runtime_loader=lambda repository: runtime,
            )
            engine.load()

            with self.assertRaisesRegex(
                DittoEngineError,
                "did not produce",
            ):
                engine.render(
                    source,
                    audio,
                    output,
                    settings=DittoRenderSettings(),
                )

            self.assertFalse(output.exists())
            self.assertIn("did not produce", engine.status()["lastError"])

    def test_requires_cuda_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            runtime = DittoRuntime(
                sdk_factory=lambda config_path, data_root: object(),
                run=lambda *args, **kwargs: None,
                seed_everything=lambda seed: None,
                cuda_available=lambda: False,
                gpu_name=lambda: "none",
            )
            engine = DittoEngine(
                config,
                runtime_loader=lambda repository: runtime,
            )

            with self.assertRaisesRegex(DittoEngineError, "CUDA"):
                engine.load()

    def test_config_requires_both_tls_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            config = DittoServiceConfig(
                **{
                    **config.__dict__,
                    "ssl_certfile": Path(directory) / "certificate.pem",
                }
            )

            with self.assertRaisesRegex(
                DittoServiceConfigError,
                "must be configured together",
            ):
                config.validate()


if __name__ == "__main__":
    unittest.main()
