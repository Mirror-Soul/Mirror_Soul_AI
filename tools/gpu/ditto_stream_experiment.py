"""Measure whether Ditto can drive a whole call as one live stream.

Instead of rendering separate idle/reply clips and stitching them, a live
call would keep one Ditto online session running: silence while the clone
listens, TTS audio while it speaks, one continuous 25 fps face stream.

This script simulates that on the GPU with real-time pacing and reports:

* speed     - can it keep up with real time (frames/s, latency drift)?
* latency   - delay from audio arriving to the matching face frame
* idle/talk - how much the face moves in silence vs. speech
* seams     - single-frame jumps, especially at silence <-> speech changes
* GPU       - peak VRAM and utilization (also with several sessions at once)

It does not touch the running Ditto services or any call.

Run inside the GPU container with the Ditto conda environment:

    cd /shareHost/C084003-ai/Mirror_Soul_AI
    /shareHost/C084003-ditto/conda-env/bin/python tools/gpu/ditto_stream_experiment.py \\
        --source /path/to/portrait.jpg

Outputs (``--out-dir``, default ``/shareHost/C084003-ai/stream-experiment``):
``stream-*.mp4`` (face + audio, to watch) and ``report.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640
DEFAULT_DITTO_DIR = Path("/shareHost/C084003-ditto/ditto-talkinghead")
DEFAULT_OUT_DIR = Path("/shareHost/C084003-ai/stream-experiment")


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    kind: str  # "silence" | "speech"
    start: float  # seconds
    end: float


@dataclass
class Timeline:
    audio: np.ndarray
    segments: list[Segment]

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def frames(self) -> int:
        return math.ceil(len(self.audio) / SAMPLES_PER_FRAME)


def quiet_noise(seconds: float, seed: int = 7) -> np.ndarray:
    """Near-silence like an idle call (about -66 dBFS), never exact zeros."""
    count = max(1, int(round(seconds * SAMPLE_RATE)))
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 16.0, count).clip(-64, 64) / 32768.0).astype(np.float32)


def build_timeline(pattern: str, speech: np.ndarray) -> Timeline:
    """``silence:3,speech,silence:4,speech:5`` -> one audio track + segments."""
    pieces: list[np.ndarray] = []
    segments: list[Segment] = []
    cursor = 0.0
    for index, token in enumerate(part.strip() for part in pattern.split(",") if part.strip()):
        kind, _, value = token.partition(":")
        if kind == "silence":
            piece = quiet_noise(float(value or 3), seed=index)
        elif kind == "speech":
            limit = len(speech) if not value else int(float(value) * SAMPLE_RATE)
            piece = speech[:limit].astype(np.float32)
        else:
            raise ValueError(f"Unknown pattern item: {token}")
        seconds = len(piece) / SAMPLE_RATE
        segments.append(Segment(kind, cursor, cursor + seconds))
        pieces.append(piece)
        cursor += seconds
    if not pieces:
        raise ValueError("Pattern is empty.")
    return Timeline(np.concatenate(pieces), segments)


# --------------------------------------------------------------------------
# probing the Ditto writer
# --------------------------------------------------------------------------


class WriterProbe:
    """Wraps Ditto's video writer: records when each frame is produced."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self.times: list[float] = []
        self.thumbs: list[np.ndarray] = []
        self.size: tuple[int, int] | None = None

    def __call__(self, frame: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        self.times.append(time.monotonic())
        if self.size is None:
            self.size = (int(frame.shape[1]), int(frame.shape[0]))
        gray = np.asarray(frame[::4, ::4], dtype=np.float32).mean(axis=2)
        self.thumbs.append(gray.astype(np.uint8))
        return self._writer(frame, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


class GpuSampler:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.memory: list[int] = []
        self.utilization: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.strip().splitlines()[0]
                memory, utilization = (int(value.strip()) for value in output.split(","))
                self.memory.append(memory)
                self.utilization.append(utilization)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        if not self.memory:
            return {"available": False}
        return {
            "available": True,
            "memoryMiBStart": self.memory[0],
            "memoryMiBPeak": max(self.memory),
            "utilizationPercentMean": round(float(np.mean(self.utilization)), 1),
            "utilizationPercentPeak": max(self.utilization),
        }


# --------------------------------------------------------------------------
# one live session
# --------------------------------------------------------------------------


@dataclass
class SessionResult:
    label: str
    started_at: float
    frame_times: list[float]
    thumbs: list[np.ndarray]
    frame_size: tuple[int, int] | None
    video_path: Path | None
    feed_lag_seconds: float
    setup_seconds: float
    gpu: dict[str, Any] = field(default_factory=dict)


def run_session(
    sdk: Any,
    timeline: Timeline,
    source: Path,
    output: Path,
    *,
    label: str,
    setup_kwargs: dict[str, Any],
    chunksize: tuple[int, int, int] = (3, 5, 2),
    realtime: bool = True,
    speech_arrival: str = "instant",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SessionResult:
    """Feed the timeline to Ditto in real time, like a live call would."""
    setup_started = clock()
    sdk.setup(str(source), str(output), **setup_kwargs)
    sdk.setup_Nd(N_d=timeline.frames)
    setup_seconds = clock() - setup_started
    probe = WriterProbe(sdk.writer)
    sdk.writer = probe

    pre, step, post = chunksize
    padded = np.concatenate([np.zeros(pre * SAMPLES_PER_FRAME, dtype=np.float32), timeline.audio])
    split_len = int(sum(chunksize) * 0.04 * SAMPLE_RATE) + 80
    worst_feed_lag = 0.0
    started = clock()
    for index in range(0, len(padded), step * SAMPLES_PER_FRAME):
        chunk = padded[index : index + split_len]
        if len(chunk) < split_len:
            chunk = np.pad(chunk, (0, split_len - len(chunk)))
        # In a call a chunk can only be sent once all of its samples
        # (including the look-ahead) have actually arrived. Silence (the
        # user talking, the clone listening) arrives in real time; a reply
        # comes from TTS all at once, so its audio is available from the
        # moment the reply starts.
        end_sample = max(0, index + split_len - pre * SAMPLES_PER_FRAME)
        arrival = started + end_sample / SAMPLE_RATE
        if speech_arrival == "instant":
            end_second = end_sample / SAMPLE_RATE
            for segment in timeline.segments:
                if segment.kind == "speech" and segment.start < end_second <= segment.end + 0.5:
                    arrival = min(arrival, started + segment.start)
        if realtime:
            wait = arrival - clock()
            if wait > 0:
                sleep(wait)
            worst_feed_lag = max(worst_feed_lag, clock() - arrival)
        sdk.run_chunk(chunk, chunksize)
    sdk.close()
    return SessionResult(
        label=label,
        started_at=started,
        frame_times=probe.times,
        thumbs=probe.thumbs,
        frame_size=probe.size,
        video_path=Path(getattr(sdk, "tmp_output_path", output)),
        feed_lag_seconds=worst_feed_lag,
        setup_seconds=setup_seconds,
    )


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 3) if values else float("nan")


def analyze(result: SessionResult, timeline: Timeline) -> dict[str, Any]:
    times = result.frame_times
    produced = len(times)
    report: dict[str, Any] = {
        "label": result.label,
        "framesExpected": timeline.frames,
        "framesProduced": produced,
        "frameSize": result.frame_size,
        "setupSeconds": round(result.setup_seconds, 3),
        "worstChunkFeedLagSeconds": round(result.feed_lag_seconds, 3),
    }
    if produced < 2:
        report["verdict"] = "NO_FRAMES"
        return report

    # Ditto's online mode spends its first audio window on warm-up and does
    # not emit those frames, so the first produced frame belongs to a later
    # point of the timeline.
    offset = max(0, timeline.frames - produced)
    report["leadingFramesSkipped"] = offset
    # Latency: frame k shows audio up to (k+1)/25 s; compare with when that
    # audio arrived in real time.
    latency = [
        times[k] - (result.started_at + (k + offset + 1) / FPS) for k in range(produced)
    ]
    onsets = []
    for segment in timeline.segments:
        if segment.kind != "speech":
            continue
        first = int(math.ceil(segment.start * FPS)) - offset
        if 0 <= first < produced:
            onsets.append(
                {
                    "speechStart": round(segment.start, 2),
                    "firstFrameAfterSeconds": round(times[first] - (result.started_at + segment.start), 3),
                }
            )
    report["speechOnset"] = onsets
    quarter = max(1, produced // 4)
    span = times[-1] - times[0]
    report["fps"] = round((produced - 1) / span, 2) if span > 0 else None
    report["latencySeconds"] = {
        "p50": _percentile(latency, 50),
        "p95": _percentile(latency, 95),
        "max": round(max(latency), 3),
        "firstQuarterMean": round(float(np.mean(latency[:quarter])), 3),
        "lastQuarterMean": round(float(np.mean(latency[-quarter:])), 3),
    }
    drift = report["latencySeconds"]["lastQuarterMean"] - report["latencySeconds"]["firstQuarterMean"]
    report["latencyDriftSeconds"] = round(drift, 3)

    # Motion and seams from 4x-downscaled gray frames.
    thumbs = result.thumbs
    diffs = np.array(
        [0.0] + [float(np.abs(thumbs[i].astype(np.int16) - thumbs[i - 1]).mean()) for i in range(1, len(thumbs))]
    )
    median = float(np.median(diffs[1:])) if len(diffs) > 1 else 0.0
    threshold = max(4.0 * median, median + 3.0)
    spikes = [
        {"second": round((i + offset) / FPS, 2), "diff": round(float(diffs[i]), 2)}
        for i in range(1, len(diffs))
        if diffs[i] > threshold and diffs[i] > 2.0 * max(diffs[i - 1], diffs[min(i + 1, len(diffs) - 1)], 0.3)
    ]
    report["motion"] = {
        "medianFrameDiff": round(median, 3),
        "seamThreshold": round(threshold, 3),
        "singleFrameJumps": spikes,
    }
    per_segment = []
    for segment in timeline.segments:
        start = int(segment.start * FPS) - offset
        end = min(len(diffs), int(segment.end * FPS) - offset)
        values = diffs[max(start, 1) : end]
        per_segment.append(
            {
                "kind": segment.kind,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "meanMotion": round(float(values.mean()), 3) if len(values) else None,
                "stillFramesPercent": round(float((values < 0.05).mean() * 100), 1) if len(values) else None,
            }
        )
    report["segments"] = per_segment

    boundaries = []
    for previous, current in zip(timeline.segments, timeline.segments[1:]):
        frame = int(current.start * FPS) - offset
        window = diffs[max(1, frame - 12) : min(len(diffs), frame + 13)]
        boundaries.append(
            {
                "second": round(current.start, 2),
                "change": f"{previous.kind}->{current.kind}",
                "maxDiffWithin0_5s": round(float(window.max()), 2) if len(window) else None,
            }
        )
    report["boundaries"] = boundaries

    keeps_up = report["fps"] is not None and report["fps"] >= FPS * 0.97 and drift < 0.5
    smooth = not spikes
    report["verdict"] = (
        "REALTIME_OK" if keeps_up and smooth
        else "REALTIME_OK_WITH_JUMPS" if keeps_up
        else "TOO_SLOW"
    )
    return report


def find_ffmpeg() -> str | None:
    candidates = [
        os.environ.get("FFMPEG_BINARY"),
        str(Path(sys.executable).with_name("ffmpeg")),
        "/shareHost/C084003-ditto/conda-env/bin/ffmpeg",
        "/opt/conda/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    from shutil import which

    return which("ffmpeg")


def mux_audio(video: Path, timeline: Timeline, output: Path, *, skipped_frames: int = 0) -> Path | None:
    """Attach the timeline audio, aligned to the first frame Ditto produced."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("  [WARN] ffmpeg를 찾지 못해 소리 없는 원본만 남깁니다 (FFMPEG_BINARY로 지정 가능).")
        return None
    wav = output.with_suffix(".wav")
    import wave

    audio = timeline.audio[skipped_frames * SAMPLES_PER_FRAME :]
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    completed = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(video), "-i", str(wav),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)],
        capture_output=True,
        text=True,
    )
    wav.unlink(missing_ok=True)
    return output if completed.returncode == 0 else None


# --------------------------------------------------------------------------
# Ditto loading and member portrait download
# --------------------------------------------------------------------------


NUMPY2_ALIASES = {
    "atan2": "arctan2",
    "atan": "arctan",
    "asin": "arcsin",
    "acos": "arccos",
    "pow": "power",
    "concat": "concatenate",
}


def add_numpy2_aliases() -> None:
    """Ditto's TensorRT helpers call NumPy 2 names (np.atan2 ...).

    The shared conda env pins NumPy 1.26 for the running services, so add
    the aliases in this process instead of upgrading NumPy.
    """
    for new, old in NUMPY2_ALIASES.items():
        if not hasattr(np, new):
            setattr(np, new, getattr(np, old))


def load_ditto(ditto_dir: Path, cfg: Path | None, data_root: Path | None) -> tuple[Any, Callable[[str], Any], str]:
    add_numpy2_aliases()
    sys.path.insert(0, str(ditto_dir))
    os.chdir(ditto_dir)
    try:
        module = importlib.import_module("stream_pipeline_online")
        module_name = "stream_pipeline_online"
    except ModuleNotFoundError:
        module = importlib.import_module("stream_pipeline_offline")
        module_name = "stream_pipeline_offline"
    if cfg is None or data_root is None:
        cfg_dir = ditto_dir / "checkpoints/ditto_cfg"
        trt_root = ditto_dir / "checkpoints/ditto_trt_Ampere_Plus"
        try:
            importlib.import_module("tensorrt")
            has_trt = trt_root.is_dir()
        except ModuleNotFoundError:
            has_trt = False
        trt_cfg = next(
            (path for path in (cfg_dir / "v0.4_hubert_cfg_trt_online.pkl", cfg_dir / "v0.4_hubert_cfg_trt.pkl") if path.is_file()),
            None,
        )
        if has_trt and trt_cfg is not None:
            cfg = cfg or trt_cfg
            data_root = data_root or trt_root
        else:
            cfg = cfg or cfg_dir / "v0.4_hubert_cfg_pytorch.pkl"
            data_root = data_root or ditto_dir / "checkpoints/ditto_pytorch"
    librosa = importlib.import_module("librosa")

    def load_audio(path: str) -> np.ndarray:
        audio, _ = librosa.core.load(path, sr=SAMPLE_RATE)
        return audio.astype(np.float32)

    sdk = module.StreamSDK(str(cfg), str(data_root if data_root.is_absolute() else ditto_dir / data_root))
    return sdk, load_audio, f"{module_name} cfg={Path(cfg).name} data={Path(data_root).name}"


def download_member_portrait(member_uuid: str, env_file: Path, out_dir: Path) -> Path:
    """Latest face-profile portrait of a member, via the face worker's S3 access."""
    for line in env_file.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
    try:
        boto3 = importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "boto3 is not installed in this Python. Run the download step with the "
            "face worker environment first:\n"
            f"  python tools/gpu/ditto_stream_experiment.py --member-uuid {member_uuid} --download-only"
        ) from exc
    bucket = os.environ.get("AWS_S3_BUCKET") or os.environ.get("DITTO_CALL_S3_BUCKET")
    if not bucket:
        raise SystemExit("AWS_S3_BUCKET is not set in the env file.")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-northeast-2"))
    profiles = [
        item
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"face-results/{member_uuid}/")
        for item in page.get("Contents", [])
        if item["Key"].endswith("face-profile.json")
    ]
    if not profiles:
        raise SystemExit(f"No face-profile.json for member {member_uuid}.")
    latest = max(profiles, key=lambda item: item["LastModified"])
    profile = json.loads(s3.get_object(Bucket=bucket, Key=latest["Key"])["Body"].read())
    portrait = profile["portrait"]
    suffix = Path(portrait["objectKey"]).suffix or ".jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"portrait-{member_uuid[:8]}{suffix}"
    s3.download_file(portrait.get("bucket", bucket), portrait["objectKey"], str(target))
    return target


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _summary_lines(report: dict[str, Any]) -> list[str]:
    if report.get("verdict") == "NO_FRAMES":
        return [f"[{report['label']}] 프레임이 나오지 않았습니다. 로그를 확인하세요."]
    if report.get("verdict") == "CAPACITY":
        return [
            f"[{report['label']}] 최대 속도 측정: {report['fps']} fps = 실시간의 {report['realtimeMultiple']}배"
            f" (이 GPU에서 이 설정으로 동시에 돌릴 수 있는 세션 수의 상한)",
            f"  프레임 {report['framesProduced']}/{report['framesExpected']} · 해상도 {report['frameSize']}",
        ]
    latency = report["latencySeconds"]
    verdict = {
        "REALTIME_OK": "실시간 가능, 튀는 프레임 없음",
        "REALTIME_OK_WITH_JUMPS": "실시간 가능, 튀는 프레임 있음",
        "TOO_SLOW": "실시간보다 느림",
    }[report["verdict"]]
    lines = [
        f"[{report['label']}] 판정: {verdict}",
        f"  속도 {report['fps']} fps (필요 25) · 프레임 {report['framesProduced']}/{report['framesExpected']} · 해상도 {report['frameSize']}",
        f"  지연: 중간값 {latency['p50']}초, 95% {latency['p95']}초, 최대 {latency['max']}초, "
        f"처음→끝 증가 {report['latencyDriftSeconds']}초",
        f"  튀는 프레임 {len(report['motion']['singleFrameJumps'])}개 (기준 {report['motion']['seamThreshold']})"
        f" · 시작 시 건너뛴 프레임 {report.get('leadingFramesSkipped', 0)}개",
    ]
    for onset in report.get("speechOnset", []):
        lines.append(
            f"  ★ 답변 음성 도착({onset['speechStart']}s) → 첫 얼굴 프레임까지 {onset['firstFrameAfterSeconds']}초"
        )
    for segment in report["segments"]:
        label = "무음" if segment["kind"] == "silence" else "음성"
        lines.append(
            f"  {segment['start']:>6.2f}~{segment['end']:<6.2f}s {label}: 움직임 {segment['meanMotion']} · 정지 프레임 {segment['stillFramesPercent']}%"
        )
    for boundary in report["boundaries"]:
        lines.append(f"  전환 {boundary['second']}s ({boundary['change']}): 전후 0.5초 최대 변화 {boundary['maxDiffWithin0_5s']}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="portrait image (or video) of the clone")
    parser.add_argument("--member-uuid", help="download that member's latest portrait from S3")
    parser.add_argument("--env-file", type=Path, default=Path("/shareHost/C084003-ai/.env.face-worker"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--speech", type=Path, help="speech audio (default: Ditto example/audio.wav)")
    parser.add_argument(
        "--pattern",
        default="silence:3,speech,silence:4,speech:4,silence:3",
        help="timeline, e.g. silence:3,speech,silence:4,speech:4",
    )
    parser.add_argument("--timesteps", default="50,25", help="sampling timesteps to compare")
    parser.add_argument("--crop-scale", type=float, default=2.3)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    parser.add_argument("--instances", type=int, default=1, help="run N sessions at the same time")
    parser.add_argument(
        "--max-speed",
        action="store_true",
        help="feed audio as fast as possible to measure the GPU's top frame rate (capacity)",
    )
    parser.add_argument("--ditto-dir", type=Path, default=DEFAULT_DITTO_DIR)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--data-root", type=Path, help="default: TensorRT engines if usable, else checkpoints/ditto_pytorch")
    parser.add_argument(
        "--speech-arrival",
        choices=("instant", "realtime"),
        default="instant",
        help="instant = reply audio arrives all at once like TTS (default); realtime = streamed like a live voice",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--instance-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.member_uuid:
        args.source = download_member_portrait(args.member_uuid, args.env_file, args.out_dir)
        print(f"[OK] 회원 초상화 다운로드: {args.source}")
        if args.download_only:
            return 0
    if args.source is None:
        print("ERROR: --source 또는 --member-uuid 가 필요합니다.", file=sys.stderr)
        return 2
    args.source = args.source.resolve()

    if args.instances > 1 and args.instance_index == 0:
        return _run_parallel(args)

    sdk, load_audio, loaded = load_ditto(args.ditto_dir, args.cfg, args.data_root)
    speech_path = args.speech.resolve() if args.speech else args.ditto_dir / "example/audio.wav"
    timeline = build_timeline(args.pattern, load_audio(str(speech_path)))
    print(f"[OK] Ditto {loaded} · 타임라인 {timeline.seconds:.1f}초 ({len(timeline.segments)}구간)")

    reports = []
    for steps in [int(value) for value in args.timesteps.split(",") if value.strip()]:
        label = f"steps{steps}" + (f"-i{args.instance_index}" if args.instances > 1 else "")
        output = args.out_dir / f"raw-{label}.mp4"
        setup_kwargs = {
            "online_mode": True,
            "crop_scale": args.crop_scale,
            "smo_k_d": args.smoothing_kernel,
            "sampling_timesteps": steps,
        }
        with GpuSampler() as gpu:
            result = run_session(
                sdk,
                timeline,
                args.source,
                output,
                label=label,
                setup_kwargs=setup_kwargs,
                speech_arrival=args.speech_arrival,
                realtime=not args.max_speed,
            )
        report = analyze(result, timeline)
        if args.max_speed:
            report["verdict"] = "CAPACITY"
            report["realtimeMultiple"] = round((report.get("fps") or 0) / FPS, 2)
        report["gpu"] = gpu.summary()
        report["speechArrival"] = args.speech_arrival
        reports.append(report)
        print("\n".join(_summary_lines(report)))
        print(f"  GPU: {report['gpu']}")
        try:
            final = mux_audio(
                result.video_path,
                timeline,
                args.out_dir / f"stream-{label}.mp4",
                skipped_frames=report.get("leadingFramesSkipped", 0),
            )
        except Exception as exc:  # the numbers above are already printed
            print(f"  [WARN] 소리 합치기 실패: {exc!r}")
            final = None
        report["video"] = str(final) if final else str(result.video_path)
        print(f"  영상: {report['video']}\n")

    report_path = args.out_dir / (f"report-i{args.instance_index}.json" if args.instances > 1 else "report.json")
    report_path.write_text(
        json.dumps(
            {
                "ditto": loaded,
                "source": str(args.source),
                "speech": str(speech_path),
                "pattern": args.pattern,
                "segments": [segment.__dict__ for segment in timeline.segments],
                "runs": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] 결과 저장: {report_path}")
    return 0


def _run_parallel(args: argparse.Namespace) -> int:
    print(f"[..] 세션 {args.instances}개를 동시에 실행합니다.")
    command = [sys.executable, str(Path(__file__).resolve())]
    passthrough = [
        "--source", str(args.source),
        "--pattern", args.pattern,
        "--timesteps", args.timesteps,
        "--crop-scale", str(args.crop_scale),
        "--smoothing-kernel", str(args.smoothing_kernel),
        "--instances", str(args.instances),
        "--ditto-dir", str(args.ditto_dir),
        "--out-dir", str(args.out_dir),
        "--speech-arrival", args.speech_arrival,
    ]
    if args.max_speed:
        passthrough.append("--max-speed")
    if args.data_root:
        passthrough += ["--data-root", str(args.data_root)]
    if args.speech:
        passthrough += ["--speech", str(args.speech.resolve())]
    if args.cfg:
        passthrough += ["--cfg", str(args.cfg)]
    processes = [
        subprocess.Popen(command + passthrough + ["--instance-index", str(index + 1)])
        for index in range(args.instances)
    ]
    codes = [process.wait() for process in processes]
    print(f"[OK] 동시 실행 종료: exit codes={codes}. report-i*.json 을 확인하세요.")
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
