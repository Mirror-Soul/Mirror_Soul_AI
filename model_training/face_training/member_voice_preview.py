import asyncio
import json
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
from shared.clone_voice import ActiveCloneVoice, find_active_clone_voice
from shared.elevenlabs_tts import (
    ElevenLabsVoiceSettings,
    synthesize_member_speech,
)


DEFAULT_MEMBER_PREVIEW_TEXT = "안녕하세요. 처음 뵙겠습니다."


@dataclass(frozen=True)
class MemberVoicePreviewResult:
    user_uuid: str
    clone_id: int
    voice_training_job_id: int | None
    text: str
    audio_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "userUuid": self.user_uuid,
            "cloneId": self.clone_id,
            "voiceTrainingJobId": self.voice_training_job_id,
            "text": self.text,
            "audioPath": str(self.audio_path),
        }


@dataclass(frozen=True)
class MemberFacePreviewResult:
    voice: MemberVoicePreviewResult
    musetalk: MuseTalkResult
    natural_motion: NaturalMotionClipResult | None = None


async def generate_member_voice_preview(
    *,
    user_uuid: str,
    clone_id: int,
    output_path: Path,
    text: str = DEFAULT_MEMBER_PREVIEW_TEXT,
    voice_resolver: Callable[..., ActiveCloneVoice] = find_active_clone_voice,
    synthesizer: Callable[..., Awaitable[bytes]] = synthesize_member_speech,
) -> MemberVoicePreviewResult:
    voice = await asyncio.to_thread(
        voice_resolver,
        user_uuid,
        expected_clone_id=clone_id,
    )
    audio = await synthesizer(
        text=text,
        voice_id=voice.elevenlabs_voice_id,
        settings=ElevenLabsVoiceSettings(
            stability=0.5,
            similarity_boost=0.9,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temporary_path.write_bytes(audio)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return MemberVoicePreviewResult(
        user_uuid=voice.user_uuid,
        clone_id=voice.clone_id,
        voice_training_job_id=voice.voice_training_job_id,
        text=text,
        audio_path=output_path,
    )


def generate_member_face_preview(
    *,
    manifest_path: Path,
    user_uuid: str,
    clone_id: int,
    text: str,
    musetalk_config: MuseTalkConfig,
    natural_motion_config: NaturalMotionConfig | None = None,
) -> MemberFacePreviewResult:
    output_root = manifest_path.parent / "outputs" / "member-voice-preview"
    voice_result = asyncio.run(
        generate_member_voice_preview(
            user_uuid=user_uuid,
            clone_id=clone_id,
            output_path=output_root / "member-voice.mp3",
            text=text,
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
