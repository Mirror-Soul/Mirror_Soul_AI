import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from model_training.face_training.musetalk_runner import (
    MuseTalkConfig,
    MuseTalkResult,
    run_musetalk_preview,
)
from model_training.face_training.natural_motion import (
    NaturalMotionClipResult,
    NaturalMotionConfig,
    prepare_natural_motion_clip,
)
from shared.elevenlabs_tts import (
    ElevenLabsVoiceSettings,
    synthesize_member_speech,
)


DEFAULT_MEMBER_PREVIEW_TEXT = "안녕하세요. 처음 뵙겠습니다."

# The face worker no longer reads the backend database. The voice for the
# preview is chosen in this order:
#   1. a member voice id passed explicitly by the caller
#   2. a fixed ElevenLabs evaluation voice (FACE_TRAINING_PREVIEW_FALLBACK_VOICE_ID)
#   3. a fixed audio file (FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH)
# Face and voice training run asynchronously, so a member voice is often not
# ready yet; in that case a fixed voice is used or the preview is skipped.
VOICE_SOURCE_MEMBER = "MEMBER_VOICE"
VOICE_SOURCE_FALLBACK_VOICE = "FALLBACK_VOICE"
VOICE_SOURCE_FALLBACK_AUDIO = "FALLBACK_AUDIO"


class MemberVoicePreviewUnavailable(RuntimeError):
    """No member or fallback voice is configured for the preview."""


@dataclass(frozen=True)
class MemberVoicePreviewResult:
    user_uuid: str
    clone_id: int
    voice_training_job_id: int | None
    text: str
    audio_path: Path
    voice_source: str = VOICE_SOURCE_MEMBER

    def to_dict(self) -> dict[str, object]:
        return {
            "userUuid": self.user_uuid,
            "cloneId": self.clone_id,
            "voiceTrainingJobId": self.voice_training_job_id,
            "text": self.text,
            "audioPath": str(self.audio_path),
            "voiceSource": self.voice_source,
        }


@dataclass(frozen=True)
class MemberFacePreviewResult:
    voice: MemberVoicePreviewResult
    musetalk: MuseTalkResult
    natural_motion: NaturalMotionClipResult | None = None


def _clean(value: str | None) -> str:
    return str(value or "").strip()


async def generate_member_voice_preview(
    *,
    user_uuid: str,
    clone_id: int,
    output_path: Path,
    text: str = DEFAULT_MEMBER_PREVIEW_TEXT,
    member_voice_id: str | None = None,
    voice_training_job_id: int | None = None,
    fallback_voice_id: str | None = None,
    fallback_audio_path: Path | str | None = None,
    synthesizer: Callable[..., Awaitable[bytes]] = synthesize_member_speech,
) -> MemberVoicePreviewResult:
    """Create the preview audio without any database lookup.

    The voice id itself is never written to the result or the manifest.
    """
    resolved_fallback_voice = _clean(
        fallback_voice_id
        if fallback_voice_id is not None
        else os.getenv("FACE_TRAINING_PREVIEW_FALLBACK_VOICE_ID")
    )
    resolved_fallback_audio = _clean(
        str(fallback_audio_path)
        if fallback_audio_path is not None
        else os.getenv("FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH")
    )

    voice_id = _clean(member_voice_id)
    voice_source = VOICE_SOURCE_MEMBER
    if not voice_id and resolved_fallback_voice:
        voice_id = resolved_fallback_voice
        voice_source = VOICE_SOURCE_FALLBACK_VOICE
        voice_training_job_id = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if voice_id:
        audio = await synthesizer(
            text=text,
            voice_id=voice_id,
            settings=ElevenLabsVoiceSettings(
                stability=0.5,
                similarity_boost=0.9,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        try:
            temporary_path.write_bytes(audio)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    elif resolved_fallback_audio:
        source = Path(resolved_fallback_audio)
        if not source.is_file():
            raise MemberVoicePreviewUnavailable(
                "FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH does not exist."
            )
        output_path = output_path.with_suffix(source.suffix or output_path.suffix)
        shutil.copyfile(source, output_path)
        voice_source = VOICE_SOURCE_FALLBACK_AUDIO
        voice_training_job_id = None
    else:
        raise MemberVoicePreviewUnavailable(
            "No member voice id was provided and no fallback preview voice "
            "(FACE_TRAINING_PREVIEW_FALLBACK_VOICE_ID or "
            "FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH) is configured."
        )

    return MemberVoicePreviewResult(
        user_uuid=user_uuid,
        clone_id=clone_id,
        voice_training_job_id=voice_training_job_id,
        text=text,
        audio_path=output_path,
        voice_source=voice_source,
    )


def generate_member_face_preview(
    *,
    manifest_path: Path,
    user_uuid: str,
    clone_id: int,
    text: str,
    musetalk_config: MuseTalkConfig,
    natural_motion_config: NaturalMotionConfig | None = None,
    member_voice_id: str | None = None,
) -> MemberFacePreviewResult:
    output_root = manifest_path.parent / "outputs" / "member-voice-preview"
    voice_result = asyncio.run(
        generate_member_voice_preview(
            user_uuid=user_uuid,
            clone_id=clone_id,
            output_path=output_root / "member-voice.mp3",
            text=text,
            member_voice_id=member_voice_id,
        )
    )
    natural_motion_result = None
    if natural_motion_config is not None:
        natural_motion_result = prepare_natural_motion_clip(
            manifest_path,
            voice_result.audio_path,
            output_root / "natural-motion-source.mp4",
            config=natural_motion_config,
        )
        source_path = natural_motion_result.clip_path
    else:
        source_path = select_source_from_manifest(manifest_path)
    musetalk_result = run_musetalk_preview(
        source_path,
        voice_result.audio_path,
        output_root / "musetalk",
        config=musetalk_config,
    )
    _record_member_face_preview(
        manifest_path,
        voice_result.to_dict(),
        musetalk_result.to_dict(),
        (
            natural_motion_result.to_dict()
            if natural_motion_result is not None
            else None
        ),
    )
    return MemberFacePreviewResult(
        voice=voice_result,
        musetalk=musetalk_result,
        natural_motion=natural_motion_result,
    )


def select_source_from_manifest(manifest_path: Path) -> Path:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[tuple[float, Path]] = []
    for video in data.get("videos", []):
        selection = video.get("frameSelection") or {}
        selected_path = selection.get("selectedSourcePath")
        if not selection.get("qualityGatePassed") or not selected_path:
            continue
        selected_frame = next(
            (
                frame
                for frame in selection.get("frames", [])
                if frame.get("path") == selected_path
            ),
            {},
        )
        candidates.append(
            (float(selected_frame.get("qualityScore", 0.0)), Path(selected_path))
        )
    if not candidates:
        raise ValueError(
            "Manifest does not contain a quality-approved face source."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _record_member_face_preview(
    manifest_path: Path,
    voice_result: dict[str, object],
    musetalk_result: dict[str, object],
    natural_motion_result: dict[str, object] | None = None,
) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["memberVoicePreview"] = {
        **voice_result,
        "museTalk": musetalk_result,
        "naturalMotion": natural_motion_result,
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
