import unittest

import numpy as np

from model_calling.realtime.audio import (
    QueuedAudioTrack,
    _QueuedAudioSegment,
    analyze_pcm_quality,
)


class RealtimeAudioQualityTests(unittest.TestCase):
    def test_flags_quiet_mostly_silent_audio(self):
        samples = np.zeros(16_000, dtype=np.int16)
        samples[:800] = 300

        quality = analyze_pcm_quality(samples.tobytes())

        self.assertAlmostEqual(quality.duration_seconds, 1.0)
        self.assertIn("LOW_VOLUME", quality.warnings)
        self.assertIn("TOO_MUCH_SILENCE", quality.warnings)

    def test_flags_clipped_audio(self):
        samples = np.full(16_000, 32767, dtype=np.int16)

        quality = analyze_pcm_quality(samples.tobytes())

        self.assertIn("CLIPPING", quality.warnings)
        self.assertGreater(quality.clipping_ratio, 0.99)

    def test_cancels_only_the_failed_turn_audio(self):
        track = QueuedAudioTrack()
        track._segments.extend(
            (
                _QueuedAudioSegment(turn_id=1, pcm=bytearray(b"first")),
                _QueuedAudioSegment(turn_id=2, pcm=bytearray(b"second")),
            )
        )

        removed = track.cancel_turn(1)

        self.assertEqual(removed, len(b"first"))
        self.assertEqual(track._queued_bytes(), len(b"second"))
        self.assertEqual(track._segments[0].turn_id, 2)


if __name__ == "__main__":
    unittest.main()
