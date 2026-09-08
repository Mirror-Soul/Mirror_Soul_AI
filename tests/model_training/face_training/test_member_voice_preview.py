import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_training.face_training.member_voice_preview import (
    MemberVoicePreviewResult,
    generate_member_face_preview,
    generate_member_voice_preview,
)
from model_training.face_training.musetalk_runner import (
    MuseTalkConfig,
    MuseTalkResult,
)
from model_training.face_training.natural_motion import (
    NaturalMotionClipResult,
    NaturalMotionConfig,
)
from shared.clone_voice import ActiveCloneVoice


class MemberVoicePreviewTests(unittest.TestCase):
    def test_uses_resolved_member_voice_without_persisting_voice_id(self) -> None:
        calls = {}

        def resolve(user_uuid: str, *, expected_clone_id: int):
            calls["resolved"] = (user_uuid, expected_clone_id)
            return ActiveCloneVoice(
                clone_id=6,
                user_uuid=user_uuid,
                voice_training_job_id=19,
                elevenlabs_voice_id="private-member-voice-id",
                status="ACTIVE",
                is_active=True,
            )

        async def synthesize(**kwargs):
            calls["voice_id"] = kwargs["voice_id"]
            return b"member-audio"

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "member.mp3"
            result = asyncio.run(
                generate_member_voice_preview(
                    user_uuid="65ebdde2-a48d-4a1c-b492-d59532a77557",
                    clone_id=6,
                    output_path=output_path,
                    voice_resolver=resolve,
                    synthesizer=synthesize,
                )
            )

            self.assertEqual(output_path.read_bytes(), b"member-audio")
            self.assertEqual(calls["voice_id"], "private-member-voice-id")
            self.assertEqual(calls["resolved"][1], 6)
            self.assertNotIn("voiceId", result.to_dict())
            self.assertNotIn("private-member-voice-id", str(result.to_dict()))

    def test_face_preview_reuses_member_audio_and_records_artifacts(self) -> None:
        user_uuid = "65ebdde2-a48d-4a1c-b492-d59532a77557"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "front.jpg"
            source_path.write_bytes(b"image")
            manifest_path = root / "preprocess-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "frameSelection": {
                                    "qualityGatePassed": True,
                                    "selectedSourcePath": str(source_path),
                                    "frames": [
                                        {
                                            "path": str(source_path),
                                            "qualityScore": 81.0,
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            audio_path = root / "member-voice.mp3"
            audio_path.write_bytes(b"member-audio")
            voice_result = MemberVoicePreviewResult(
                user_uuid=user_uuid,
                clone_id=6,
                voice_training_job_id=19,
                text="hello",
                audio_path=audio_path,
            )
            output_path = root / "member-face.mp4"
            output_path.write_bytes(b"video")
            musetalk_result = MuseTalkResult(
                source_path=source_path,
                audio_path=audio_path,
                output_path=output_path,
                log_path=root / "musetalk.log",
                inference_config_path=root / "inference-config.json",
                bbox_shift=0,
            )
            config = MuseTalkConfig(
                repository_dir=root,
                python_binary=root / "python",
            )

            with patch(
                "model_training.face_training.member_voice_preview."
                "generate_member_voice_preview",
                return_value=voice_result,
            ) as generate_voice:
                with patch(
                    "model_training.face_training.member_voice_preview."
                    "run_musetalk_preview",
                    return_value=musetalk_result,
                ) as run_musetalk:
                    result = generate_member_face_preview(
                        manifest_path=manifest_path,
                        user_uuid=user_uuid,
                        clone_id=6,
                        text="hello",
                        musetalk_config=config,
                    )

            self.assertEqual(result.musetalk.output_path, output_path)
            self.assertEqual(generate_voice.call_args.kwargs["clone_id"], 6)
            self.assertEqual(run_musetalk.call_args.args[:2], (source_path, audio_path))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded = manifest["memberVoicePreview"]
            self.assertEqual(recorded["cloneId"], 6)
            self.assertEqual(recorded["audioPath"], str(audio_path))
            self.assertEqual(recorded["museTalk"]["outputPath"], str(output_path))
            self.assertNotIn("voiceId", recorded)

    def test_uses_natural_motion_video_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "preprocess-manifest.json"
            manifest_path.write_text('{"videos": []}', encoding="utf-8")
            audio_path = root / "member.mp3"
            clip_path = root / "natural-motion.mp4"
            voice_result = MemberVoicePreviewResult(
                user_uuid="member",
                clone_id=4,
                voice_training_job_id=4,
                text="hello",
                audio_path=audio_path,
            )
            motion_result = NaturalMotionClipResult(
                source_video_path=root / "source.mov",
                clip_path=clip_path,
                start_seconds=1.0,
                duration_seconds=3.0,
                score=80.0,
                front_frame_ratio=1.0,
                confidence="high",
            )
            musetalk_result = MuseTalkResult(
                source_path=clip_path,
                audio_path=audio_path,
                output_path=root / "result.mp4",
                log_path=root / "musetalk.log",
                inference_config_path=root / "inference-config.json",
                bbox_shift=0,
            )

            with patch(
                "model_training.face_training.member_voice_preview."
                "generate_member_voice_preview",
                return_value=voice_result,
            ), patch(
                "model_training.face_training.member_voice_preview."
                "prepare_natural_motion_clip",
                return_value=motion_result,
            ), patch(
                "model_training.face_training.member_voice_preview."
                "run_musetalk_preview",
                return_value=musetalk_result,
            ) as run_musetalk:
                result = generate_member_face_preview(
                    manifest_path=manifest_path,
                    user_uuid="member",
                    clone_id=4,
                    text="hello",
                    musetalk_config=MuseTalkConfig(root, root / "python"),
                    natural_motion_config=NaturalMotionConfig(),
                )

            self.assertEqual(run_musetalk.call_args.args[0], clip_path)
            self.assertEqual(result.natural_motion, motion_result)
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                recorded["memberVoicePreview"]["naturalMotion"]["confidence"],
                "high",
            )


if __name__ == "__main__":
    unittest.main()
