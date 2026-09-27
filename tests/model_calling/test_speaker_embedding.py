import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_calling.clone_similarity import speaker_embedding
from model_calling.clone_similarity.speaker_embedding import (
    SpeakerSimilarityAudio,
    SpeakerSimilarityUnavailable,
)


class SpeakerEmbeddingAudioTests(unittest.TestCase):
    def test_normalizes_m4a_to_mono_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.m4a"
            output = root / "sample.wav"
            source.write_bytes(b"m4a")

            def create_output(command, **kwargs):
                Path(command[-1]).write_bytes(b"wav")
                return Mock(returncode=0)

            with patch.object(
                speaker_embedding.subprocess,
                "run",
                side_effect=create_output,
            ) as run:
                result = speaker_embedding._normalize_audio_file(
                    source,
                    output,
                    target_sample_rate=16000,
                )

        self.assertEqual(result, output)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")

    def test_reports_missing_ffmpeg_without_audio_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private-member-sample.m4a"
            source.write_bytes(b"m4a")

            with patch.object(
                speaker_embedding.subprocess,
                "run",
                side_effect=FileNotFoundError,
            ):
                with self.assertRaises(SpeakerSimilarityUnavailable) as context:
                    speaker_embedding._normalize_audio_file(
                        source,
                        root / "sample.wav",
                        target_sample_rate=16000,
                    )

        self.assertIn("ffmpeg is required", str(context.exception))
        self.assertNotIn("private-member-sample", str(context.exception))

    def test_evaluation_normalizes_original_and_clone_audio(self) -> None:
        classifier = SimpleNamespace(hparams=SimpleNamespace(sample_rate=16000))
        normalized_paths = []

        def normalize(source_path, output_path, *, target_sample_rate):
            output_path.write_bytes(b"wav")
            normalized_paths.append((source_path, output_path, target_sample_rate))
            return output_path

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "clone-reference.mp3"
            reference.write_bytes(b"mp3")
            with (
                patch.object(
                    speaker_embedding,
                    "_get_classifier",
                    return_value=classifier,
                ),
                patch.object(
                    speaker_embedding,
                    "_normalize_audio_file",
                    side_effect=normalize,
                ),
                patch.object(
                    speaker_embedding,
                    "_average_embeddings",
                    return_value="original",
                ),
                patch.object(
                    speaker_embedding,
                    "_encode_file",
                    return_value="clone",
                ) as encode,
                patch.object(
                    speaker_embedding,
                    "_cosine_similarity",
                    return_value=0.5,
                ),
                patch.dict(
                    "os.environ",
                    {"CLONE_SIMILARITY_ENABLE_SPEAKER_EMBEDDING": "true"},
                ),
            ):
                result = speaker_embedding.evaluate_speaker_similarity(
                    original_audios=[
                        SpeakerSimilarityAudio(
                            filename="onboarding.m4a",
                            content=b"m4a",
                            content_type="audio/mp4",
                        )
                    ],
                    elevenlabs_voice_id="voice-id",
                    reference_audio_path=reference,
                )

        self.assertEqual(len(normalized_paths), 2)
        self.assertTrue(str(normalized_paths[0][1]).endswith(".wav"))
        self.assertTrue(str(normalized_paths[1][1]).endswith(".wav"))
        self.assertTrue(str(encode.call_args.args[1]).endswith(".wav"))
        self.assertEqual(result.cosine_similarity, 0.5)


if __name__ == "__main__":
    unittest.main()
