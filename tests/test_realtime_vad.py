"""Utterance detection: short noises must not open endless recordings."""

import asyncio
import os
import unittest
from fractions import Fraction
from unittest.mock import patch

import av
import numpy as np
from aiortc.mediastreams import MediaStreamError

from model_calling.realtime.audio import receive_utterances

RATE = 16_000
FRAME = 320  # 20 ms


class _Track:
    def __init__(self, segments: list[tuple[float, int]]) -> None:
        samples = []
        rng = np.random.default_rng(1)
        for seconds, level in segments:
            count = int(seconds * RATE)
            if level:
                tone = level * np.sin(np.arange(count) * 2 * np.pi * 220 / RATE)
            else:
                tone = rng.normal(0, 20, count)
            samples.append(tone)
        self.pcm = np.concatenate(samples).astype(np.int16)
        self.position = 0

    async def recv(self):
        if self.position >= len(self.pcm):
            raise MediaStreamError
        chunk = self.pcm[self.position : self.position + FRAME]
        self.position += FRAME
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.time_base = Fraction(1, RATE)
        frame.pts = self.position
        return frame


class _Output:
    is_playing = False


def _run(segments: list[tuple[float, int]], **env: str) -> list[float]:
    utterances: list[float] = []

    async def collect(wav: bytes) -> None:
        utterances.append((len(wav) - 44) / (RATE * 2))

    settings = {
        "REALTIME_VAD_STARTUP_GRACE_SECONDS": "0",
        "REALTIME_VAD_ENERGY_THRESHOLD": "900",
        "REALTIME_VAD_SILENCE_SECONDS": "0.8",
        "REALTIME_VAD_MIN_SPEECH_SECONDS": "0.7",
        "REALTIME_VAD_MAX_SPEECH_SECONDS": "15",
        **env,
    }
    with patch.dict(os.environ, settings):
        asyncio.run(receive_utterances(_Track(segments), _Output(), collect))
    return utterances


class VadTests(unittest.TestCase):
    def test_normal_sentence_is_one_utterance(self) -> None:
        result = _run([(0.5, 0), (2.0, 5000), (1.5, 0)])
        self.assertEqual(len(result), 1)
        self.assertLess(result[0], 3.5)

    def test_short_noise_then_long_silence_is_discarded(self) -> None:
        # Previously: a 0.2 s click opened an utterance that kept every
        # following second of silence until real speech came, then sent a
        # 30+ second clip to STT.
        result = _run([(0.2, 5000), (30.0, 0), (1.5, 5000), (1.2, 0)])
        self.assertEqual(len(result), 1)
        self.assertLess(result[0], 3.5)

    def test_total_length_is_capped_even_with_pauses(self) -> None:
        segments = []
        for _ in range(20):
            segments += [(0.3, 5000), (0.6, 0)]
        result = _run(segments + [(1.0, 0)], REALTIME_VAD_MAX_UTTERANCE_SECONDS="8")
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(all(length <= 8.5 for length in result))


if __name__ == "__main__":
    unittest.main()
