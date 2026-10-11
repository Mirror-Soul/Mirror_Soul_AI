import importlib.util
from pathlib import Path
import struct
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "gpu" / "benchmark-ditto-workers.py"
SPEC = importlib.util.spec_from_file_location("benchmark_ditto_workers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class DittoWorkerBenchmarkTests(unittest.TestCase):
    def test_parser_handles_split_frame_packets(self) -> None:
        frame = b"jpeg"
        payload = (
            benchmark.STREAM_MAGIC
            + struct.pack(">I", len(frame))
            + frame
            + struct.pack(">I", 0)
        )
        parser = benchmark._FrameParser()
        frames = []
        for chunk in (payload[:3], payload[3:8], payload[8:11], payload[11:]):
            frames.extend(parser.feed(chunk))
        parser.finish()

        self.assertEqual(frames, [frame])

    def test_quality_thresholds_surface_slow_streams(self) -> None:
        self.assertEqual(benchmark._stream_quality(800, 28.0), "PASS")
        self.assertEqual(benchmark._stream_quality(1700, 25.0), "WARNING")
        self.assertEqual(benchmark._stream_quality(800, 22.0), "WARNING")
        self.assertEqual(benchmark._stream_quality(3100, 30.0), "FAIL")
        self.assertEqual(benchmark._stream_quality(900, 19.0), "FAIL")


if __name__ == "__main__":
    unittest.main()
