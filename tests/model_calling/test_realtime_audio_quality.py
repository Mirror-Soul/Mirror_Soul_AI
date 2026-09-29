import unittest

import numpy as np

from model_calling.realtime.audio import analyze_pcm_quality


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


if __name__ == "__main__":
    unittest.main()
