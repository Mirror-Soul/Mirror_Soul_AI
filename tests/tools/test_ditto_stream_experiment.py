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


class _SkippingSdk(_FakeSdk):
    """Drops the first 10 frames like Ditto's online warm-up window."""

    def run_chunk(self, chunk, chunksize):
        start = self.index
        self.index += chunksize[1]
        for k in range(start, start + chunksize[1]):
            if k >= 10:
                self.writer(np.full((32, 32, 3), 100 + (k % 2), dtype=np.uint8), fmt="rgb")


class OnsetTests(unittest.TestCase):
    def test_instant_speech_arrival_and_skipped_frames(self) -> None:
        timeline = build_timeline("silence:0.4,speech:0.4,silence:0.2", np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1)
        clock = [0.0]
        fed_at = []

        def fake_clock():
            return clock[0]

        def fake_sleep(seconds):
            clock[0] += seconds

        sdk = _SkippingSdk(delay=0.0)
        original = sdk.run_chunk

        def record(chunk, chunksize):
            fed_at.append(clock[0])
            original(chunk, chunksize)

        sdk.run_chunk = record
        result = run_session(
            sdk, timeline, Path("p.jpg"), Path("o.mp4"), label="t",
            setup_kwargs={}, clock=fake_clock, sleep=fake_sleep,
        )
        # speech (0.4 s .. 0.8 s) is available at 0.4 s, so feeding never waits past it
        self.assertLessEqual(max(fed_at[:6]), 0.4 + 1e-6)
        report = analyze(result, timeline)
        self.assertEqual(report["leadingFramesSkipped"], timeline.frames - len(result.frame_times))
        self.assertEqual(len(report["speechOnset"]), 1)
