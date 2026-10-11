import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import av
import numpy as np

from model_calling.realtime.diagnostics import (
    ReplyVideoDiagnosticCapture,
    ReplyVideoDiagnosticConfig,
    _prune_diagnostics,
    _write_reply_video,
)


def _jpeg_frame() -> bytes:
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mjpeg")
    stream = container.add_stream("mjpeg", rate=25)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuvj420p"
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :, 1] = 160
    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


def _wav_audio() -> bytes:
    import wave

    output = io.BytesIO()
    samples = np.zeros(16_000, dtype="<i2")
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


class ReplyVideoDiagnosticTests(unittest.TestCase):
    def test_disabled_capture_keeps_no_frames(self) -> None:
        capture = ReplyVideoDiagnosticCapture(
            ReplyVideoDiagnosticConfig(),
            audio_bytes=b"audio",
            call_id=1,
            turn_id=2,
        )
        capture.add_frame(b"frame")
        capture.finish_in_background()
        self.assertEqual(capture.frames, [])

    def test_buffer_limit_discards_partial_capture(self) -> None:
        capture = ReplyVideoDiagnosticCapture(
            ReplyVideoDiagnosticConfig(
                output_dir=Path("diagnostics"),
                max_buffer_bytes=5,
            ),
            audio_bytes=b"audio",
            call_id=1,
            turn_id=2,
        )
        capture.add_frame(b"1234")
        capture.add_frame(b"56")
        self.assertTrue(capture.overflowed)
        self.assertEqual(capture.frames, [])

    def test_writer_combines_numbered_frames_and_tts_audio(self) -> None:
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path("tmp")) as directory:
            output_dir = Path(directory)
            config = ReplyVideoDiagnosticConfig(output_dir=output_dir)
            frame = _jpeg_frame()
            _write_reply_video(
                config,
                (frame,) * 25,
                _wav_audio(),
                17,
                3,
                25,
            )

            outputs = list(output_dir.glob("call-17-turn-3-*.mp4"))
            self.assertEqual(len(outputs), 1)
            container = av.open(str(outputs[0]))
            try:
                self.assertTrue(any(stream.type == "video" for stream in container.streams))
                self.assertTrue(any(stream.type == "audio" for stream in container.streams))
                self.assertEqual(len(list(container.decode(video=0))), 25)
            finally:
                container.close()

            audio_container = av.open(str(outputs[0]))
            try:
                audio_frames = list(audio_container.decode(audio=0))
                self.assertTrue(audio_frames)
                audio_seconds = sum(frame.samples for frame in audio_frames) / 48_000
                self.assertAlmostEqual(audio_seconds, 1.0, delta=0.05)
            finally:
                audio_container.close()

    def test_background_write_does_not_block_realtime_path(self) -> None:
        capture = ReplyVideoDiagnosticCapture(
            ReplyVideoDiagnosticConfig(output_dir=Path("diagnostics")),
            audio_bytes=b"audio",
            call_id=1,
            turn_id=2,
        )
        capture.add_frame(b"frame")
        fake_task = Mock()
        with patch(
            "model_calling.realtime.diagnostics.asyncio.create_task",
            return_value=fake_task,
        ) as create_task:
            task = capture.finish_in_background()

        self.assertIs(task, fake_task)
        coroutine = create_task.call_args.args[0]
        coroutine.close()
        fake_task.add_done_callback.assert_called_once()

    def test_config_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = ReplyVideoDiagnosticConfig.from_env()
        self.assertIsNone(config.output_dir)

    def test_prune_removes_expired_and_excess_files(self) -> None:
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path("tmp")) as directory:
            output_dir = Path(directory)
            old = output_dir / "call-1-turn-1-old.mp4"
            recent = [
                output_dir / f"call-1-turn-{index}-recent.mp4"
                for index in range(2, 5)
            ]
            old.write_bytes(b"old")
            for index, path in enumerate(recent):
                path.write_bytes(b"recent")
                os.utime(path, (1_000 + index, 1_000 + index))
            os.utime(old, (1, 1))

            with patch(
                "model_calling.realtime.diagnostics.time.time",
                return_value=2_000,
            ):
                _prune_diagnostics(output_dir, max_files=2, max_age_hours=0.5)

            self.assertFalse(old.exists())
            self.assertEqual(len(list(output_dir.glob("*.mp4"))), 2)


if __name__ == "__main__":
    unittest.main()
