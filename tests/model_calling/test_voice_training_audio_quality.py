import io
import math
import os
import shutil
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from model_calling.voice_training.audio_quality import (
    RejectedVoiceTrainingAudio,
    ValidatedVoiceTrainingAudio,
    VoiceTrainingAudioBatch,
    VoiceTrainingAudioMetrics,
    VoiceTrainingAudioQualityConfig,
    VoiceTrainingAudioSample,
    VoiceTrainingInputQualityError,
    analyze_normalized_wav,
    normalize_and_validate_voice_samples,
)

os.environ.setdefault("OPENAI_API_KEY", "test-only")

from model_calling.voice_training.worker import (
    DownloadedAudio,
    VoiceTrainingWorkerError,
    _prepare_voice_training_audio,
)


def _wav_bytes(
    *,
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
    amplitude: int = 9000,
    frequency: float = 220.0,
) -> bytes:
    sample_count = int(duration_seconds * sample_rate)
    samples = array(
        "h",
        (
            int(amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            for index in range(sample_count)
        ),
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())
    return output.getvalue()


def _write_input_as_normalized_output(
    sample: VoiceTrainingAudioSample,
    *,
    sample_number: int,
    workspace: Path,
    config: VoiceTrainingAudioQualityConfig,
) -> Path:
    del config
    output_path = workspace / f"normalized-{sample_number:02d}.wav"
    output_path.write_bytes(sample.content)
    return output_path


class VoiceTrainingAudioMetricsTests(unittest.TestCase):
    def test_analyzes_clean_speech_like_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clean.wav"
            path.write_bytes(_wav_bytes())

            metrics = analyze_normalized_wav(path)

        self.assertAlmostEqual(metrics.duration_seconds, 2.0, places=2)
        self.assertGreater(metrics.rms_dbfs, -20.0)
        self.assertLess(metrics.peak_dbfs, -5.0)
        self.assertEqual(metrics.silence_ratio, 0.0)
        self.assertEqual(metrics.clipping_ratio, 0.0)

    def test_detects_silence_and_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            silent_path = Path(temp_dir) / "silent.wav"
            silent_path.write_bytes(_wav_bytes(amplitude=0))
            clipped_path = Path(temp_dir) / "clipped.wav"
            clipped_path.write_bytes(_wav_bytes(amplitude=32767, frequency=1000.0))

            silent = analyze_normalized_wav(silent_path)
            clipped = analyze_normalized_wav(clipped_path)

        self.assertEqual(silent.rms_dbfs, -120.0)
        self.assertEqual(silent.silence_ratio, 1.0)
        self.assertGreater(clipped.clipping_ratio, 0.01)


class VoiceTrainingAudioBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VoiceTrainingAudioQualityConfig(
            min_accepted_samples=2,
            min_batch_duration_seconds=3.0,
        )

    @patch(
        "model_calling.voice_training.audio_quality._normalize_sample",
        side_effect=_write_input_as_normalized_output,
    )
    def test_keeps_only_valid_normalized_samples(self, _normalize) -> None:
        samples = [
            VoiceTrainingAudioSample("one.m4a", _wav_bytes(), "audio/mp4"),
            VoiceTrainingAudioSample("two.webm", _wav_bytes(), "audio/webm"),
            VoiceTrainingAudioSample(
                "silent.wav",
                _wav_bytes(amplitude=0),
                "audio/wav",
            ),
        ]

        batch = normalize_and_validate_voice_samples(samples, config=self.config)

        self.assertEqual(len(batch.accepted), 2)
        self.assertEqual(batch.accepted[0].filename, "voice-sample-01.wav")
        self.assertEqual(batch.accepted[0].content_type, "audio/wav")
        self.assertEqual(batch.total_duration_seconds, 4.0)
        self.assertEqual(len(batch.rejected), 1)
        self.assertEqual(batch.rejected[0].sample_number, 3)
        self.assertIn("audio_too_quiet", batch.rejected[0].reason_codes)
        self.assertIn("too_much_silence", batch.rejected[0].reason_codes)

    @patch(
        "model_calling.voice_training.audio_quality._normalize_sample",
        side_effect=_write_input_as_normalized_output,
    )
    def test_rejects_batch_with_too_few_valid_samples(self, _normalize) -> None:
        samples = [
            VoiceTrainingAudioSample("one.wav", _wav_bytes(), "audio/wav"),
            VoiceTrainingAudioSample(
                "silent.wav",
                _wav_bytes(amplitude=0),
                "audio/wav",
            ),
        ]

        with self.assertRaises(VoiceTrainingInputQualityError) as raised:
            normalize_and_validate_voice_samples(samples, config=self.config)

        self.assertEqual(raised.exception.code, "insufficient_valid_samples")
        self.assertEqual(len(raised.exception.batch.accepted), 1)
        self.assertEqual(len(raised.exception.batch.rejected), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_real_ffmpeg_normalization_produces_standard_wav(self) -> None:
        config = VoiceTrainingAudioQualityConfig(
            ffmpeg_binary=str(shutil.which("ffmpeg")),
            min_accepted_samples=1,
            min_batch_duration_seconds=1.0,
        )

        batch = normalize_and_validate_voice_samples(
            [VoiceTrainingAudioSample("input.wav", _wav_bytes(), "audio/wav")],
            config=config,
        )

        self.assertEqual(len(batch.accepted), 1)
        with wave.open(io.BytesIO(batch.accepted[0].content), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 16000)


class VoiceTrainingWorkerPreparationTests(unittest.TestCase):
    @patch("model_calling.voice_training.worker._env_bool", return_value=True)
    @patch("model_calling.voice_training.worker.normalize_and_validate_voice_samples")
    def test_worker_uses_only_normalized_accepted_audio(
        self,
        normalize,
        _env_bool,
    ) -> None:
        metrics = VoiceTrainingAudioMetrics(2.0, -12.0, -3.0, 0.1, 0.0)
        normalize.return_value = VoiceTrainingAudioBatch(
            accepted=(
                ValidatedVoiceTrainingAudio(
                    sample_number=1,
                    filename="voice-sample-01.wav",
                    content=b"normalized",
                    content_type="audio/wav",
                    metrics=metrics,
                ),
            ),
            rejected=(RejectedVoiceTrainingAudio(2, ("audio_too_quiet",)),),
            total_duration_seconds=2.0,
        )

        prepared = _prepare_voice_training_audio(
            [DownloadedAudio("source.m4a", b"source", "audio/mp4")],
            job_id=17,
        )

        self.assertEqual(
            prepared,
            [DownloadedAudio("voice-sample-01.wav", b"normalized", "audio/wav")],
        )

    @patch("model_calling.voice_training.worker._env_bool", return_value=True)
    @patch("model_calling.voice_training.worker.normalize_and_validate_voice_samples")
    def test_worker_turns_batch_quality_failure_into_job_failure(
        self,
        normalize,
        _env_bool,
    ) -> None:
        batch = VoiceTrainingAudioBatch(
            accepted=(),
            rejected=(RejectedVoiceTrainingAudio(1, ("decode_failed",)),),
            total_duration_seconds=0.0,
        )
        normalize.side_effect = VoiceTrainingInputQualityError(
            "insufficient_valid_samples",
            batch,
        )

        with self.assertRaisesRegex(
            VoiceTrainingWorkerError,
            "voice_input_quality_failed",
        ):
            _prepare_voice_training_audio(
                [DownloadedAudio("source.m4a", b"source", "audio/mp4")],
                job_id=18,
            )


if __name__ == "__main__":
    unittest.main()
