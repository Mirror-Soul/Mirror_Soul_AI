from __future__ import annotations

import asyncio
import io
import math
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from typing import Any, Iterator

import av
import numpy as np
from aiortc import MediaStreamTrack


VIDEO_CLOCK_RATE = 90_000


class QueuedVideoTrackError(RuntimeError):
    pass


@dataclass
class _VideoSegment:
    container: Any
    frames: Iterator[av.VideoFrame]
    segment_id: int = 0

    def close(self) -> None:
        self.container.close()


def _open_segment(video_bytes: bytes, *, segment_id: int = 0) -> _VideoSegment:
    container = None
    try:
        container = av.open(io.BytesIO(video_bytes))
        if not any(stream.type == "video" for stream in container.streams):
            raise QueuedVideoTrackError(
                "Rendered response does not contain a video stream."
            )
        decoded_frames = iter(container.decode(video=0))
        try:
            first_frame = next(decoded_frames)
        except StopIteration as exc:
            raise QueuedVideoTrackError(
                "Rendered response contains no decodable video frames."
            ) from exc
        return _VideoSegment(
            container=container,
            frames=iter(chain((first_frame,), decoded_frames)),
            segment_id=segment_id,
        )
    except Exception:
        if container is not None:
            container.close()
        raise


class QueuedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        *,
        width: int = 540,
        height: int = 960,
        fps: int = 25,
        idle_motion_enabled: bool = True,
        idle_motion_scale: float = 0.012,
        idle_motion_period_seconds: float = 6.0,
        transition_frames: int = 0,
        stream_buffer_max_frames: int = 125,
        stream_max_lag_frames: int = 3,
    ) -> None:
        super().__init__()
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("Video dimensions must be positive even numbers.")
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        if idle_motion_scale < 0 or idle_motion_scale > 0.05:
            raise ValueError("Idle motion scale must be between 0 and 0.05.")
        if idle_motion_period_seconds <= 0:
            raise ValueError("Idle motion period must be positive.")
        if transition_frames < 0 or transition_frames > 50:
            raise ValueError("Transition frames must be between 0 and 50.")
        if stream_buffer_max_frames <= 0:
            raise ValueError("Stream frame buffer limit must be positive.")
        if stream_max_lag_frames < 0 or stream_max_lag_frames > 50:
            raise ValueError("Stream max lag frames must be between 0 and 50.")

        self.width = width
        self.height = height
        self.fps = fps
        self.idle_motion_enabled = idle_motion_enabled
        self.idle_motion_scale = idle_motion_scale
        self.idle_motion_period_seconds = idle_motion_period_seconds
        self._segments: deque[_VideoSegment] = deque()
        self._active_segment: _VideoSegment | None = None
        self._idle_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_number = 0
        self._started_at: float | None = None
        self._sent_video_frames = 0
        self.transition_frames = transition_frames
        self._next_segment_id = 1
        # Idle loop: a Ditto clip rendered from silence that starts and ends
        # on the neutral portrait pose, replayed from its first frame
        # whenever the track returns to idle.
        self._idle_video_bytes: bytes | None = None
        self._idle_loop: _VideoSegment | None = None
        self._idle_loop_generation = 0
        # Crossfade state: the last frame sent and the frame being faded out.
        self._last_source: tuple[str, int] | None = None
        self._last_output: av.VideoFrame | None = None
        self._blend_from: np.ndarray | None = None
        self._blend_step = 0
        self._transitions = 0
        self._stream_frames: deque[tuple[int, bytes]] = deque()
        self._stream_buffer_max_frames = stream_buffer_max_frames
        self._stream_next_id = 1
        self._stream_pending_id: int | None = None
        self._stream_active_id: int | None = None
        self._stream_finished = False
        self._stream_ready = asyncio.Event()
        self._stream_space = asyncio.Event()
        self._stream_space.set()
        # Lip sync: the stream can become ready before reply audio is decoded
        # and queued. The clock is therefore armed separately when the audio
        # queue reports when playback will start. Frame N belongs at that
        # audio start time + N / fps. When frames arrive late (GPU shared with
        # another call, slow tunnel), old frames are skipped so the mouth
        # catches up instead of staying late for the rest of the reply.
        # 0 disables catching up.
        self.stream_max_lag_frames = stream_max_lag_frames
        self._stream_activated_at: float | None = None
        self._stream_consumed = 0
        self._stream_dropped = 0
        self._stream_stalled = 0
        self._stream_max_lag = 0
        self._stream_skip_logged_at = 0.0

    @property
    def is_playing(self) -> bool:
        return (
            self._stream_active_id is not None
            or self._active_segment is not None
            or bool(self._segments)
        )

    @property
    def has_idle_video(self) -> bool:
        return self._idle_video_bytes is not None

    @property
    def transition_count(self) -> int:
        return self._transitions

    def set_idle_image(self, image_bytes: bytes) -> None:
        if not image_bytes:
            raise QueuedVideoTrackError("Idle portrait must not be empty.")
        container = None
        try:
            container = av.open(io.BytesIO(image_bytes))
            frame = next(container.decode(video=0))
            self._idle_rgb = frame.reformat(
                width=self.width,
                height=self.height,
                format="rgb24",
            ).to_ndarray()
        except Exception as exc:
            raise QueuedVideoTrackError(
                f"Unable to decode idle portrait: {exc}"
            ) from exc
        finally:
            if container is not None:
                container.close()

    def set_idle_video(self, video_bytes: bytes) -> None:
        """Use a rendered idle clip instead of the still portrait while idle."""
        if not video_bytes:
            raise QueuedVideoTrackError("Idle video must not be empty.")
        probe = _open_segment(video_bytes)
        probe.close()
        self._close_idle_loop()
        self._idle_video_bytes = video_bytes
        self._idle_loop_generation += 1
        print(
            "[VIDEO_OUT] idle loop video ready: "
            f"encoded_bytes={len(video_bytes)}",
            flush=True,
        )

    def enqueue_encoded_video(self, video_bytes: bytes) -> None:
        if not video_bytes:
            raise QueuedVideoTrackError("Rendered video must not be empty.")
        segment = _open_segment(video_bytes, segment_id=self._next_segment_id)
        self._next_segment_id += 1
        self._segments.append(segment)

        print(
            "[VIDEO_OUT] queued Ditto video: "
            f"encoded_bytes={len(video_bytes)} segments={len(self._segments)}",
            flush=True,
        )

    def begin_frame_stream(self) -> int:
        if self._stream_active_id is not None:
            raise QueuedVideoTrackError("A Ditto frame stream is already active.")
        self._stream_frames.clear()
        self._stream_activated_at = None
        stream_id = self._stream_next_id
        self._stream_next_id += 1
        self._stream_pending_id = stream_id
        self._stream_finished = False
        self._stream_ready.clear()
        self._stream_space.set()
        return stream_id

    async def enqueue_stream_frame(self, stream_id: int, frame_bytes: bytes) -> bool:
        if not frame_bytes:
            raise QueuedVideoTrackError("Stream frame must not be empty.")
        while len(self._stream_frames) >= self._stream_buffer_max_frames:
            if stream_id not in {self._stream_pending_id, self._stream_active_id}:
                return False
            self._stream_space.clear()
            if len(self._stream_frames) < self._stream_buffer_max_frames:
                self._stream_space.set()
                break
            await self._stream_space.wait()
        if stream_id not in {self._stream_pending_id, self._stream_active_id}:
            return False
        self._stream_frames.append((stream_id, frame_bytes))
        self._stream_ready.set()
        return True

    async def wait_for_stream_buffer(
        self,
        stream_id: int,
        min_frames: int,
        timeout_seconds: float,
    ) -> int:
        async def wait() -> int:
            while True:
                count = sum(1 for item_id, _ in self._stream_frames if item_id == stream_id)
                if count >= min_frames or self._stream_finished:
                    return count
                if stream_id != self._stream_pending_id:
                    return 0
                self._stream_ready.clear()
                count = sum(
                    1 for item_id, _ in self._stream_frames if item_id == stream_id
                )
                if count >= min_frames or self._stream_finished:
                    continue
                await self._stream_ready.wait()

        return await asyncio.wait_for(wait(), timeout=timeout_seconds)

    def activate_frame_stream(self, stream_id: int) -> int:
        if stream_id != self._stream_pending_id:
            raise QueuedVideoTrackError("Ditto frame stream is no longer pending.")
        count = sum(1 for item_id, _ in self._stream_frames if item_id == stream_id)
        if count <= 0:
            raise QueuedVideoTrackError("Ditto frame stream has no buffered frames.")
        self._stream_pending_id = None
        self._stream_active_id = stream_id
        self._stream_activated_at = None
        self._stream_consumed = 0
        self._stream_dropped = 0
        self._stream_stalled = 0
        self._stream_max_lag = 0
        print(
            "[VIDEO_OUT] Ditto frame stream activated: "
            f"stream_id={stream_id} buffered_frames={count}",
            flush=True,
        )
        return count

    def synchronize_frame_stream(self, start_delay_seconds: float = 0.0) -> bool:
        """Start the lip-sync clock at the matching audio playback time."""
        if not math.isfinite(start_delay_seconds) or start_delay_seconds < 0:
            raise QueuedVideoTrackError(
                "Audio playback delay must be a finite non-negative number."
            )
        stream_id = self._stream_active_id
        if stream_id is None:
            return False
        self._stream_activated_at = time.monotonic() + start_delay_seconds
        print(
            "[VIDEO_OUT] Ditto frame stream synchronized with audio: "
            f"stream_id={stream_id} "
            f"start_delay_ms={round(start_delay_seconds * 1000)}",
            flush=True,
        )
        return True

    def finish_frame_stream(self, stream_id: int) -> None:
        if stream_id in {self._stream_pending_id, self._stream_active_id}:
            self._stream_finished = True
            self._stream_ready.set()

    def fail_frame_stream(self, stream_id: int) -> None:
        if stream_id == self._stream_pending_id:
            self._stream_frames = deque(
                item for item in self._stream_frames if item[0] != stream_id
            )
            self._stream_pending_id = None
        if stream_id == self._stream_active_id:
            self._stream_finished = True
        self._stream_ready.set()
        self._stream_space.set()

    async def recv(self) -> av.VideoFrame:
        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            target_time = self._started_at + (self._frame_number / self.fps)
            await asyncio.sleep(max(0.0, target_time - time.monotonic()))

        stream_id = self._stream_active_id
        frame = self._next_stream_frame()
        if frame is not None:
            source = ("reply-stream", stream_id or 0)
        else:
            frame = self._next_rendered_frame()
            if frame is not None:
                assert self._active_segment is not None
                source = ("reply", self._active_segment.segment_id)
            else:
                if (
                    self._last_source is not None
                    and self._last_source[0] in {"reply", "reply-stream"}
                ):
                    # Replies end on the neutral pose, and so does the idle
                    # loop's first frame: restart the loop there.
                    self._close_idle_loop()
                frame, source = self._next_idle_frame()
        frame = self._apply_transition(frame, source)
        frame = frame.reformat(
            width=self.width,
            height=self.height,
            format="yuv420p",
        )
        self._last_output = frame
        frame.pts = round(self._frame_number * VIDEO_CLOCK_RATE / self.fps)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        self._frame_number += 1
        self._sent_video_frames += 1
        if self._sent_video_frames == 1 or self._sent_video_frames % 125 == 0:
            print(
                "[VIDEO_OUT] sending video frame: "
                f"frames={self._sent_video_frames} playing={self.is_playing}",
                flush=True,
            )
        return frame

    def stop(self) -> None:
        self._close_active_segment()
        self._close_idle_loop()
        while self._segments:
            self._segments.popleft().close()
        self._stream_frames.clear()
        self._stream_pending_id = None
        self._stream_active_id = None
        self._stream_activated_at = None
        self._stream_finished = True
        self._stream_ready.set()
        self._stream_space.set()
        super().stop()

    def _next_stream_frame(self) -> av.VideoFrame | None:
        stream_id = self._stream_active_id
        if stream_id is None:
            return None
        audio_started_at = self._stream_activated_at
        if audio_started_at is None or time.monotonic() < audio_started_at:
            return None
        while self._stream_frames and self._stream_frames[0][0] != stream_id:
            self._stream_frames.popleft()
        if self._stream_frames:
            self._skip_late_stream_frames()
            _, frame_bytes = self._stream_frames.popleft()
            self._stream_consumed += 1
            self._stream_space.set()
            container = None
            try:
                container = av.open(io.BytesIO(frame_bytes))
                return next(container.decode(video=0))
            except Exception as exc:
                print(
                    "[VIDEO_OUT] streamed frame decode failed: "
                    f"stream_id={stream_id} error={exc!r}",
                    flush=True,
                )
                if self._last_output is not None:
                    return av.VideoFrame.from_ndarray(
                        self._last_output.to_ndarray(format="rgb24"),
                        format="rgb24",
                    )
                return None
            finally:
                if container is not None:
                    container.close()
        if self._stream_finished:
            self._stream_active_id = None
            self._stream_activated_at = None
            self._stream_finished = False
            self._stream_space.set()
            print(
                f"[VIDEO_OUT] Ditto frame stream completed: stream_id={stream_id} "
                f"frames={self._stream_consumed} dropped={self._stream_dropped} "
                f"stalled={self._stream_stalled} "
                f"max_lag_ms={round(self._stream_max_lag * 1000 / self.fps)}",
                flush=True,
            )
            return None
        # Frames are late: hold the last picture; audio keeps playing.
        self._stream_stalled += 1
        if self._last_output is not None:
            return av.VideoFrame.from_ndarray(
                self._last_output.to_ndarray(format="rgb24"),
                format="rgb24",
            )
        return None

    def _stream_lag_frames(self) -> int:
        """How many frames the stream is behind the reply audio right now."""
        if self._stream_activated_at is None:
            return 0
        expected = int((time.monotonic() - self._stream_activated_at) * self.fps)
        return expected - self._stream_consumed

    def _skip_late_stream_frames(self) -> None:
        lag = self._stream_lag_frames()
        self._stream_max_lag = max(self._stream_max_lag, lag)
        if self.stream_max_lag_frames <= 0 or lag <= self.stream_max_lag_frames:
            return
        # Drop buffered frames that are already past due, but always keep
        # one to show now. Dropped frames are never decoded.
        skipped = 0
        while (
            skipped < lag
            and len(self._stream_frames) > 1
            and self._stream_frames[1][0] == self._stream_active_id
        ):
            self._stream_frames.popleft()
            skipped += 1
        if skipped:
            self._stream_consumed += skipped
            self._stream_dropped += skipped
            now = time.monotonic()
            if now - self._stream_skip_logged_at < 1.0:
                return
            self._stream_skip_logged_at = now
            print(
                "[VIDEO_OUT] stream behind audio, skipped late frames: "
                f"stream_id={self._stream_active_id} skipped={skipped} "
                f"lag_ms={round(lag * 1000 / self.fps)}",
                flush=True,
            )

    def _next_rendered_frame(self) -> av.VideoFrame | None:
        while True:
            if self._active_segment is None:
                if not self._segments:
                    return None
                self._active_segment = self._segments.popleft()
            try:
                return next(self._active_segment.frames)
            except StopIteration:
                self._close_active_segment()
                print("[VIDEO_OUT] Ditto video segment completed", flush=True)
            except Exception as exc:
                self._close_active_segment()
                print(
                    "[VIDEO_OUT] Ditto video segment decode failed; "
                    f"returning to idle: error={exc!r}",
                    flush=True,
                )
                return None

    def _next_idle_frame(self) -> tuple[av.VideoFrame, tuple[str, int]]:
        if self._idle_video_bytes is not None:
            for _ in range(2):
                if self._idle_loop is None:
                    try:
                        self._idle_loop = _open_segment(self._idle_video_bytes)
                    except Exception as exc:
                        self._disable_idle_video(exc)
                        break
                    self._idle_loop_generation += 1
                try:
                    frame = next(self._idle_loop.frames)
                    return frame, ("idle-loop", self._idle_loop_generation)
                except StopIteration:
                    self._close_idle_loop()
                except Exception as exc:
                    self._disable_idle_video(exc)
                    break
        return self._idle_frame(), ("idle-portrait", 0)

    def _disable_idle_video(self, exc: Exception) -> None:
        self._close_idle_loop()
        self._idle_video_bytes = None
        print(
            "[VIDEO_OUT] idle loop video failed; using still portrait: "
            f"error={exc!r}",
            flush=True,
        )

    def _apply_transition(
        self,
        frame: av.VideoFrame,
        source: tuple[str, int],
    ) -> av.VideoFrame:
        previous_source = self._last_source
        self._last_source = source
        if self.transition_frames <= 0:
            return frame
        if (
            previous_source is not None
            and source != previous_source
            and self._last_output is not None
        ):
            self._blend_from = self._last_output.to_ndarray(format="rgb24")
            self._blend_step = 0
            if previous_source[0] != source[0]:
                # Idle-loop wraps also blend, but are not worth a log line.
                self._transitions += 1
                print(
                    "[VIDEO_OUT] crossfade started: "
                    f"from={previous_source[0]} to={source[0]} "
                    f"frames={self.transition_frames}",
                    flush=True,
                )
        if self._blend_from is None:
            return frame

        self._blend_step += 1
        alpha = self._blend_step / (self.transition_frames + 1)
        current = frame.reformat(
            width=self.width,
            height=self.height,
            format="rgb24",
        ).to_ndarray()
        blended = (
            current.astype(np.float32) * alpha
            + self._blend_from.astype(np.float32) * (1.0 - alpha)
        )
        if self._blend_step >= self.transition_frames:
            self._blend_from = None
        return av.VideoFrame.from_ndarray(
            np.clip(blended + 0.5, 0, 255).astype(np.uint8),
            format="rgb24",
        )

    def _close_idle_loop(self) -> None:
        if self._idle_loop is not None:
            self._idle_loop.close()
            self._idle_loop = None

    def _idle_frame(self) -> av.VideoFrame:
        if not self.idle_motion_enabled or self.idle_motion_scale == 0:
            idle_rgb = self._idle_rgb
        else:
            elapsed = self._frame_number / self.fps
            phase = 2 * math.pi * elapsed / self.idle_motion_period_seconds
            motion = (1 - math.cos(phase)) / 2
            scale = 1 + self.idle_motion_scale * motion
            crop_width = max(2, min(self.width, round(self.width / scale)))
            crop_height = max(2, min(self.height, round(self.height / scale)))
            horizontal_shift = round(
                self.width * self.idle_motion_scale * 0.15 * math.sin(phase * 0.7)
            )
            vertical_shift = round(
                self.height * self.idle_motion_scale * 0.2 * math.sin(phase)
            )
            left = max(
                0,
                min(
                    self.width - crop_width,
                    (self.width - crop_width) // 2 + horizontal_shift,
                ),
            )
            top = max(
                0,
                min(
                    self.height - crop_height,
                    (self.height - crop_height) // 2 + vertical_shift,
                ),
            )
            idle_rgb = self._idle_rgb[
                top : top + crop_height,
                left : left + crop_width,
            ]

        return av.VideoFrame.from_ndarray(idle_rgb, format="rgb24")

    def _close_active_segment(self) -> None:
        if self._active_segment is not None:
            self._active_segment.close()
            self._active_segment = None
