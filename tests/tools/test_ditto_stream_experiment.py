import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tools.gpu.ditto_stream_experiment import (
    FPS,
    SAMPLE_RATE,
    analyze,
    build_timeline,
    run_session,
)


class _Writer:
    def __init__(self) -> None:
        self.frames = 0
        self.closed = False

    def __call__(self, frame, fmt="rgb"):
        self.frames += 1

    def close(self):
        self.closed = True


class _FakeSdk:
    """Emits 5 frames per chunk on a worker thread, like Ditto's online mode."""

    def __init__(self, delay: float, jump_at: int | None = None) -> None:
        self.delay = delay
        self.jump_at = jump_at
        self.calls = []

    def setup(self, source, output, **kwargs):
        self.kwargs = kwargs
        self.writer = _Writer()
        self.tmp_output_path = output
        self.index = 0
        self.threads = []

    def setup_Nd(self, N_d):
        self.N_d = N_d

    def run_chunk(self, chunk, chunksize):
        self.calls.append(len(chunk))

        def emit(start):
            time.sleep(self.delay)
            for k in range(start, start + chunksize[1]):
                level = 120 if (self.jump_at is not None and k >= self.jump_at) else 100
                frame = np.full((32, 32, 3), level + (k % 2), dtype=np.uint8)
                self.writer(frame, fmt="rgb")

        thread = threading.Thread(target=emit, args=(self.index,))
        thread.start()
        thread.join()
        self.index += chunksize[1]

    def close(self):
        self.writer.close()


class ExperimentTests(unittest.TestCase):
    def test_timeline_pattern(self) -> None:
        speech = np.ones(SAMPLE_RATE * 2, dtype=np.float32) * 0.1
        timeline = build_timeline("silence:1,speech,silence:0.5,speech:1", speech)
        self.assertEqual([s.kind for s in timeline.segments], ["silence", "speech", "silence", "speech"])
        self.assertAlmostEqual(timeline.seconds, 4.5)
        self.assertEqual(timeline.frames, -(-len(timeline.audio) // (SAMPLE_RATE // FPS)))
        self.assertLess(np.abs(timeline.audio[: SAMPLE_RATE]).max(), 0.01)

    def test_session_feeds_chunks_and_measures_frames(self) -> None:
        timeline = build_timeline("silence:0.4,speech:0.4", np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1)
        sdk = _FakeSdk(delay=0.0, jump_at=10)
        result = run_session(
            sdk, timeline, Path("p.jpg"), Path("o.mp4"),
            label="t", setup_kwargs={"online_mode": True}, realtime=False,
        )
        self.assertTrue(sdk.kwargs["online_mode"])
        self.assertTrue(sdk.writer.closed if hasattr(sdk.writer, "closed") else True)
        report = analyze(result, timeline)
        self.assertGreaterEqual(report["framesProduced"], timeline.frames)
        jumps = report["motion"]["singleFrameJumps"]
        self.assertEqual([j["second"] for j in jumps], [10 / FPS])
        self.assertEqual(report["boundaries"][0]["change"], "silence->speech")


if __name__ == "__main__":
    unittest.main()
