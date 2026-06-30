import asyncio
import io
import os
import time
import wave
from collections import deque
from fractions import Fraction
from typing import Awaitable, Callable

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError


OUTPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLES_PER_FRAME = 960
INPUT_SAMPLE_RATE = 16000


class QueuedAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._pcm = bytearray()
        self._pts = 0
        self._started_at: float | None = None
        self._sent_audio_frames = 0
        self._sent_silence_frames = 0
        self._last_silence_log_at = 0.0

    @property
    def is_playing(self) -> bool:
        return bool(self._pcm)

    def enqueue_encoded_audio(self, audio_bytes: bytes) -> None:
        pcm_before = len(self._pcm)
        decoded_frames = 0
        resampled_frames = 0
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=OUTPUT_SAMPLE_RATE,
        )

        try:
            for frame in container.decode(audio=0):
                decoded_frames += 1
                for resampled in resampler.resample(frame):
                    resampled_frames += 1
                    self._pcm.extend(resampled.to_ndarray().tobytes())

            for resampled in resampler.resample(None):
                resampled_frames += 1
                self._pcm.extend(resampled.to_ndarray().tobytes())
        finally:
            container.close()

        added_pcm = len(self._pcm) - pcm_before
        duration_seconds = added_pcm / (OUTPUT_SAMPLE_RATE * 2)
        print(
            "[AUDIO_OUT] queued encoded audio: "
            f"encoded_bytes={len(audio_bytes)} decoded_frames={decoded_frames} "
            f"resampled_frames={resampled_frames} pcm_added={added_pcm} "
            f"pcm_total={len(self._pcm)} duration={duration_seconds:.2f}s",
            flush=True,
        )

    async def recv(self) -> av.AudioFrame:
        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            target_time = self._started_at + (self._pts / OUTPUT_SAMPLE_RATE)
            await asyncio.sleep(max(0.0, target_time - time.monotonic()))

        byte_count = OUTPUT_SAMPLES_PER_FRAME * 2
        if len(self._pcm) >= byte_count:
            pcm = bytes(self._pcm[:byte_count])
            del self._pcm[:byte_count]
            self._sent_audio_frames += 1
            if self._sent_audio_frames == 1 or self._sent_audio_frames % 50 == 0:
                print(
                    "[AUDIO_OUT] sending audio frame: "
                    f"frames={self._sent_audio_frames} remaining_pcm={len(self._pcm)}",
                    flush=True,
                )
        else:
            pcm = bytes(byte_count)
            self._sent_silence_frames += 1
            now = time.monotonic()
            if now - self._last_silence_log_at >= 5.0:
                self._last_silence_log_at = now
                print(
                    "[AUDIO_OUT] sending silence frame heartbeat: "
                    f"silence_frames={self._sent_silence_frames}",
                    flush=True,
                )

        samples = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = OUTPUT_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, OUTPUT_SAMPLE_RATE)
        self._pts += OUTPUT_SAMPLES_PER_FRAME
        return frame


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


async def receive_utterances(
    track: MediaStreamTrack,
    output_track: QueuedAudioTrack,
    on_utterance: Callable[[bytes], Awaitable[None]],
) -> None:
    energy_threshold = int(os.getenv("REALTIME_VAD_ENERGY_THRESHOLD", "450"))
    silence_seconds = float(os.getenv("REALTIME_VAD_SILENCE_SECONDS", "0.8"))
    min_speech_seconds = float(os.getenv("REALTIME_VAD_MIN_SPEECH_SECONDS", "0.4"))
    max_speech_seconds = float(os.getenv("REALTIME_VAD_MAX_SPEECH_SECONDS", "15"))

    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=INPUT_SAMPLE_RATE,
    )
    pre_roll: deque[bytes] = deque(maxlen=8)
    utterance = bytearray()
    speech_seconds = 0.0
    silence_accumulated = 0.0
    speaking = False
    received_frames = 0
    last_input_log_at = 0.0
    last_playback_skip_log_at = 0.0

    print(
        "[AUDIO_IN] VAD config: "
        f"threshold={energy_threshold} silence={silence_seconds}s "
        f"min_speech={min_speech_seconds}s max_speech={max_speech_seconds}s",
        flush=True,
    )

    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            print("[REALTIME] incoming audio track ended", flush=True)
            return

        received_frames += 1
        now = time.monotonic()
        if now - last_input_log_at >= 5.0:
            last_input_log_at = now
            print(
                "[AUDIO_IN] receiving frames heartbeat: "
                f"frames={received_frames} output_playing={output_track.is_playing}",
                flush=True,
            )

        if output_track.is_playing:
            if now - last_playback_skip_log_at >= 2.0:
                last_playback_skip_log_at = now
                print(
                    "[AUDIO_IN] input ignored while AI audio is playing",
                    flush=True,
                )
            pre_roll.clear()
            utterance.clear()
            speech_seconds = 0.0
            silence_accumulated = 0.0
            speaking = False
            continue

        for audio_frame in resampler.resample(frame):
            pcm = audio_frame.to_ndarray().tobytes()
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                continue

            duration = samples.size / INPUT_SAMPLE_RATE
            rms = float(np.sqrt(np.mean(np.square(samples))))
            has_voice = rms >= energy_threshold

            if not speaking:
                pre_roll.append(pcm)
                if not has_voice:
                    continue

                speaking = True
                print(
                    "[AUDIO_IN] speech started: "
                    f"rms={rms:.1f} threshold={energy_threshold} "
                    f"pre_roll_chunks={len(pre_roll)}",
                    flush=True,
                )
                for chunk in pre_roll:
                    utterance.extend(chunk)
                pre_roll.clear()
            else:
                utterance.extend(pcm)

            if has_voice:
                speech_seconds += duration
                silence_accumulated = 0.0
            else:
                silence_accumulated += duration

            reached_silence = (
                speech_seconds >= min_speech_seconds
                and silence_accumulated >= silence_seconds
            )
            reached_limit = speech_seconds >= max_speech_seconds
            if reached_silence or reached_limit:
                wav_bytes = pcm_to_wav_bytes(bytes(utterance))
                reason = "max_speech" if reached_limit else "silence"
                print(
                    "[AUDIO_IN] utterance queued: "
                    f"reason={reason} speech={speech_seconds:.2f}s "
                    f"silence={silence_accumulated:.2f}s wav_bytes={len(wav_bytes)}",
                    flush=True,
                )
                await on_utterance(wav_bytes)
                utterance.clear()
                speech_seconds = 0.0
                silence_accumulated = 0.0
                speaking = False
