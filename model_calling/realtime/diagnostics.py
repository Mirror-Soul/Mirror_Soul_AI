from __future__ import annotations

import asyncio
import io
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av


_diagnostic_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class ReplyVideoDiagnosticConfig:
    """Opt-in capture of generated clone video with its exact TTS audio."""

    output_dir: Path | None = None
    max_files: int = 20
    max_buffer_bytes: int = 64 * 1024 * 1024
    max_age_hours: float = 24.0
    max_pending_tasks: int = 2

    @classmethod
    def from_env(cls) -> "ReplyVideoDiagnosticConfig":
        value = os.getenv("DITTO_CALL_DIAGNOSTIC_DIR", "").strip()
        return cls(
            output_dir=Path(value).expanduser() if value else None,
            max_files=int(os.getenv("DITTO_CALL_DIAGNOSTIC_MAX_FILES", "20")),
            max_buffer_bytes=int(
                os.getenv(
                    "DITTO_CALL_DIAGNOSTIC_MAX_BUFFER_BYTES",
                    str(64 * 1024 * 1024),
                )
            ),
            max_age_hours=float(
                os.getenv("DITTO_CALL_DIAGNOSTIC_MAX_AGE_HOURS", "24")
            ),
            max_pending_tasks=int(
                os.getenv("DITTO_CALL_DIAGNOSTIC_MAX_PENDING_TASKS", "2")
            ),
        )

    def validate(self) -> None:
        if self.max_files <= 0 or self.max_files > 200:
            raise ValueError("Diagnostic max files must be between 1 and 200.")
        if (
            self.max_buffer_bytes <= 0
            or self.max_buffer_bytes > 512 * 1024 * 1024
        ):
            raise ValueError(
                "Diagnostic frame buffer must be between 1 byte and 512 MiB."
            )
        if self.max_age_hours <= 0 or self.max_age_hours > 168:
            raise ValueError(
                "Diagnostic max age must be greater than 0 and at most 168 hours."
            )
        if self.max_pending_tasks <= 0 or self.max_pending_tasks > 8:
            raise ValueError(
                "Diagnostic pending task limit must be between 1 and 8."
            )


class ReplyVideoDiagnosticCapture:
    """Buffers one generated reply and writes it outside the realtime path."""

    def __init__(
        self,
        config: ReplyVideoDiagnosticConfig,
        *,
        audio_bytes: bytes,
        call_id: int | None,
        turn_id: int | None,
        fps: int = 25,
    ) -> None:
        config.validate()
        self.config = config
        self.audio_bytes = audio_bytes
        self.call_id = call_id
        self.turn_id = turn_id
        self.fps = fps
        self.frames: list[bytes] = []
        self.buffered_bytes = 0
        self.overflowed = False

    @property
    def enabled(self) -> bool:
        return self.config.output_dir is not None

    def add_frame(self, frame_bytes: bytes) -> None:
        if not self.enabled or self.overflowed:
            return
        next_size = self.buffered_bytes + len(frame_bytes)
        if next_size > self.config.max_buffer_bytes:
            self.frames.clear()
            self.buffered_bytes = 0
            self.overflowed = True
            return
        self.frames.append(frame_bytes)
        self.buffered_bytes = next_size

    def finish_in_background(self) -> asyncio.Task[None] | None:
        if not self.enabled:
            return None
        if self.overflowed:
            print(
                "[DITTO_DIAGNOSTIC] capture skipped: frame buffer limit exceeded "
                f"call_id={self.call_id} turn_id={self.turn_id}",
                flush=True,
            )
            return None
        if not self.frames or not self.audio_bytes:
            return None
        if len(_diagnostic_tasks) >= self.config.max_pending_tasks:
            self.frames.clear()
            self.buffered_bytes = 0
            print(
                "[DITTO_DIAGNOSTIC] capture skipped: encoder backlog limit "
                f"reached pending={len(_diagnostic_tasks)} "
                f"call_id={self.call_id} turn_id={self.turn_id}",
                flush=True,
            )
            return None

        frames = tuple(self.frames)
        self.frames.clear()
        task = asyncio.create_task(
            asyncio.to_thread(
                _write_reply_video,
                self.config,
                frames,
                self.audio_bytes,
                self.call_id,
                self.turn_id,
                self.fps,
            )
        )
        _diagnostic_tasks.add(task)
        task.add_done_callback(_diagnostic_task_done)
        return task


def _diagnostic_task_done(task: asyncio.Task[None]) -> None:
    _diagnostic_tasks.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        print(
            f"[DITTO_DIAGNOSTIC] capture failed: error={error!r}",
            flush=True,
        )


def _write_reply_video(
    config: ReplyVideoDiagnosticConfig,
    frames: tuple[bytes, ...],
    audio_bytes: bytes,
    call_id: int | None,
    turn_id: int | None,
    fps: int,
) -> None:
    output_dir = config.output_dir
    if output_dir is None:
        return
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(output_dir, 0o700)
    except OSError:
        pass

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    call_label = str(call_id) if call_id is not None else "unknown"
    turn_label = str(turn_id) if turn_id is not None else "unknown"
    output_path = output_dir / (
        f"call-{call_label}-turn-{turn_label}-{timestamp}.mp4"
    )

    temporary_path = output_path.with_suffix(".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    try:
        _mux_reply_video(temporary_path, frames, audio_bytes, fps)
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise RuntimeError("Diagnostic MP4 encoder produced no output.")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise

    try:
        os.chmod(output_path, 0o600)
    except OSError:
        pass
    _prune_diagnostics(
        output_dir,
        config.max_files,
        config.max_age_hours,
    )
    print(
        "[DITTO_DIAGNOSTIC] reply video saved: "
        f"call_id={call_id} turn_id={turn_id} frames={len(frames)} "
        f"path={output_path}",
        flush=True,
    )


def _mux_reply_video(
    output_path: Path,
    frames: tuple[bytes, ...],
    audio_bytes: bytes,
    fps: int,
) -> None:
    first_frame = _decode_image(frames[0])
    output = av.open(str(output_path), mode="w", format="mp4")
    audio_input = None
    try:
        video_stream = output.add_stream("libx264", rate=fps)
        video_stream.width = first_frame.width
        video_stream.height = first_frame.height
        video_stream.pix_fmt = "yuv420p"
        video_time_base = Fraction(1, fps)

        audio_input = av.open(io.BytesIO(audio_bytes))
        input_stream = next(
            (stream for stream in audio_input.streams if stream.type == "audio"),
            None,
        )
        if input_stream is None:
            raise RuntimeError("Diagnostic TTS payload has no audio stream.")
        # Register every MP4 stream before muxing the first packet. Adding an
        # audio stream after video muxing is unsupported by some FFmpeg builds.
        sample_rate = 48_000
        layout = "mono"
        audio_stream = output.add_stream("aac", rate=sample_rate)
        audio_stream.layout = layout
        audio_time_base = Fraction(1, sample_rate)
        audio_stream.time_base = audio_time_base
        resampler = av.AudioResampler(
            format="fltp",
            layout=layout,
            rate=sample_rate,
        )

        for index, encoded_frame in enumerate(frames):
            frame = first_frame if index == 0 else _decode_image(encoded_frame)
            frame = frame.reformat(
                width=video_stream.width,
                height=video_stream.height,
                format="yuv420p",
            )
            frame.pts = index
            frame.time_base = video_time_base
            for packet in video_stream.encode(frame):
                output.mux(packet)
        for packet in video_stream.encode(None):
            output.mux(packet)

        audio_pts = 0
        for decoded in audio_input.decode(audio=0):
            for resampled in resampler.resample(decoded):
                resampled.pts = audio_pts
                resampled.time_base = audio_time_base
                audio_pts += resampled.samples
                for packet in audio_stream.encode(resampled):
                    output.mux(packet)
        for resampled in resampler.resample(None):
            resampled.pts = audio_pts
            resampled.time_base = audio_time_base
            audio_pts += resampled.samples
            for packet in audio_stream.encode(resampled):
                output.mux(packet)
        for packet in audio_stream.encode(None):
            output.mux(packet)
    finally:
        if audio_input is not None:
            audio_input.close()
        output.close()


def _decode_image(content: bytes) -> av.VideoFrame:
    container = av.open(io.BytesIO(content))
    try:
        return next(container.decode(video=0))
    finally:
        container.close()


def _prune_diagnostics(
    output_dir: Path,
    max_files: int,
    max_age_hours: float,
) -> None:
    try:
        cutoff = time.time() - max_age_hours * 3600
        paths = sorted(
            output_dir.glob("call-*-turn-*.mp4"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        paths = [path for path in paths if path.exists()]
        for path in paths[max_files:]:
            path.unlink(missing_ok=True)
    except OSError as exc:
        print(
            f"[DITTO_DIAGNOSTIC] old capture cleanup skipped: error={exc!r}",
            flush=True,
        )
