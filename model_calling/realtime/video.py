from __future__ import annotations

import asyncio
import io
import math
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
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

    def close(self) -> None:
        self.container.close()


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

    @property
    def is_playing(self) -> bool:
        return self._active_segment is not None or bool(self._segments)

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

    def enqueue_encoded_video(self, video_bytes: bytes) -> None:
        if not video_bytes:
            raise QueuedVideoTrackError("Rendered video must not be empty.")
        container = None
        try:
            container = av.open(io.BytesIO(video_bytes))
            if not any(stream.type == "video" for stream in container.streams):
                raise QueuedVideoTrackError(
                    "Rendered response does not contain a video stream."
                )
            segment = _VideoSegment(
                container=container,
                frames=iter(container.decode(video=0)),
            )
            self._segments.append(segment)
        except Exception:
            if container is not None:
                container.close()
            raise

        print(
            "[VIDEO_OUT] queued Ditto video: "
            f"encoded_bytes={len(video_bytes)} segments={len(self._segments)}",
            flush=True,
        )

    async def recv(self) -> av.VideoFrame:
        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            target_time = self._started_at + (self._frame_number / self.fps)
            await asyncio.sleep(max(0.0, target_time - time.monotonic()))

        frame = self._next_rendered_frame()
        if frame is None:
            frame = self._idle_frame()
        frame = frame.reformat(
            width=self.width,
            height=self.height,
            format="yuv420p",
        )
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
        while self._segments:
            self._segments.popleft().close()
        super().stop()

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
