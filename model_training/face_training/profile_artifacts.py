from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_training.face_training.member_voice_preview import (
    select_source_from_manifest,
)
from model_training.face_training.message import FaceTrainingMessage


class FaceProfileArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceProfileArtifacts:
    bucket: str
    prefix: str
    profile_key: str
    portrait_key: str
    manifest_key: str
    preview_key: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "profileKey": self.profile_key,
            "portraitKey": self.portrait_key,
            "manifestKey": self.manifest_key,
            "previewKey": self.preview_key,
        }


def upload_face_profile_artifacts(
    s3_client: Any,
    *,
    message: FaceTrainingMessage,
    manifest_path: Path,
    result_prefix: str = "face-results",
    engine: str = "ditto",
    engine_version: str = "v0.4-hubert-pytorch",
    crop_scale: float = 2.3,
    smoothing_kernel: int = 5,
    sampling_timesteps: int = 50,
) -> FaceProfileArtifacts:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FaceProfileArtifactError(f"Manifest not found: {manifest_path}")

    normalized_prefix = _normalize_prefix(result_prefix)
    _validate_render_settings(
        crop_scale=crop_scale,
        smoothing_kernel=smoothing_kernel,
        sampling_timesteps=sampling_timesteps,
    )
    source_path = select_source_from_manifest(manifest_path).resolve()
    if not source_path.is_file():
        raise FaceProfileArtifactError(f"Selected portrait not found: {source_path}")

    artifact_prefix = (
        f"{normalized_prefix}/{message.user_uuid}/job-{message.job_id}"
    )
    portrait_suffix = source_path.suffix.lower() or ".jpg"
    portrait_key = f"{artifact_prefix}/portrait{portrait_suffix}"
    manifest_key = f"{artifact_prefix}/preprocess-manifest.json"
    profile_key = f"{artifact_prefix}/face-profile.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preview_path = _find_preview_path(manifest)
    preview_key = (
        f"{artifact_prefix}/preview{preview_path.suffix.lower()}"
        if preview_path is not None
        else None
    )

    _put_file(
        s3_client,
        bucket=message.bucket,
        key=portrait_key,
        path=source_path,
        content_type=mimetypes.guess_type(source_path.name)[0] or "image/jpeg",
    )
    _put_file(
        s3_client,
        bucket=message.bucket,
        key=manifest_key,
        path=manifest_path,
        content_type="application/json",
    )
    if preview_path is not None and preview_key is not None:
        _put_file(
            s3_client,
            bucket=message.bucket,
            key=preview_key,
            path=preview_path,
            content_type="video/mp4",
        )

    profile = {
        "schemaVersion": 1,
        "profileType": "AUDIO_DRIVEN_FACE",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "jobId": message.job_id,
        "userUuid": message.user_uuid,
        "cloneId": message.clone_id,
        "engine": {
            "name": engine,
            "version": engine_version,
            "renderSettings": {
                "cropScale": crop_scale,
                "smoothingKernel": smoothing_kernel,
                "samplingTimesteps": sampling_timesteps,
            },
        },
        "portrait": {
            "bucket": message.bucket,
            "objectKey": portrait_key,
            "contentType": mimetypes.guess_type(source_path.name)[0]
            or "image/jpeg",
        },
        "preprocessManifest": {
            "bucket": message.bucket,
            "objectKey": manifest_key,
        },
        "preview": (
            {"bucket": message.bucket, "objectKey": preview_key}
            if preview_key is not None
            else None
        ),
        "quality": _selected_quality(manifest, source_path),
    }
    s3_client.put_object(
        Bucket=message.bucket,
        Key=profile_key,
        Body=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return FaceProfileArtifacts(
        bucket=message.bucket,
        prefix=artifact_prefix,
        profile_key=profile_key,
        portrait_key=portrait_key,
        manifest_key=manifest_key,
        preview_key=preview_key,
    )


def _normalize_prefix(value: str) -> str:
    normalized = value.strip().strip("/").replace("\\", "/")
    if not normalized or ".." in normalized.split("/"):
        raise FaceProfileArtifactError("Invalid face result prefix.")
    return normalized


def _validate_render_settings(
    *,
    crop_scale: float,
    smoothing_kernel: int,
    sampling_timesteps: int,
) -> None:
    if crop_scale <= 0:
        raise FaceProfileArtifactError("Ditto crop scale must be positive.")
    if smoothing_kernel <= 0 or smoothing_kernel % 2 == 0:
        raise FaceProfileArtifactError(
            "Ditto smoothing kernel must be a positive odd integer."
        )
    if sampling_timesteps <= 0:
        raise FaceProfileArtifactError(
            "Ditto sampling timesteps must be positive."
        )


def _put_file(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    path: Path,
    content_type: str,
) -> None:
    with path.open("rb") as body:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )


def _find_preview_path(manifest: dict[str, Any]) -> Path | None:
    candidates = [
        ((manifest.get("memberVoicePreview") or {}).get("museTalk") or {}).get(
            "outputPath"
        ),
        (manifest.get("livePortrait") or {}).get("outputPath"),
    ]
    for value in candidates:
        if value:
            path = Path(str(value)).resolve()
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def _selected_quality(
    manifest: dict[str, Any],
    source_path: Path,
) -> dict[str, object] | None:
    source = str(source_path)
    for video in manifest.get("videos", []):
        selection = video.get("frameSelection") or {}
        for frame in selection.get("frames", []):
            try:
                frame_path = str(Path(str(frame.get("path"))).resolve())
            except (OSError, RuntimeError):
                continue
            if frame_path != source:
                continue
            return {
                "qualityScore": frame.get("qualityScore"),
                "sharpness": frame.get("sharpness"),
                "brightness": frame.get("brightness"),
                "contrast": frame.get("contrast"),
                "faceCoverage": frame.get("faceCoverage"),
                "centerOffset": frame.get("centerOffset"),
                "view": frame.get("view"),
                "qualityTier": selection.get("qualityTier", "NORMAL"),
                "selectionMode": selection.get("selectionMode", "STRICT"),
                "qualityWarnings": selection.get("qualityWarnings", []),
            }
    return None
