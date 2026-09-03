from __future__ import annotations

import argparse
import os
import time
from pathlib import PurePosixPath
from uuid import UUID

from dotenv import load_dotenv

from model_training.face_training.message import FaceTrainingMessage
from model_training.face_training.worker import (
    FaceTrainingWorkerError,
    _boto3_client,
    _preprocess_face_training_message,
)

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a local face avatar preview from existing S3 videos "
            "without consuming SQS messages or uploading results."
        )
    )
    parser.add_argument(
        "--object-key",
        action="append",
        required=True,
        dest="object_keys",
        help="S3 face video object key. Repeat to provide multiple videos.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("AWS_S3_BUCKET", "mirrorsoul-storage-64"),
        help="S3 bucket containing the face videos.",
    )
    parser.add_argument(
        "--user-uuid",
        help="User UUID. Inferred from face-videos/<uuid>/... when omitted.",
    )
    parser.add_argument("--clone-id", type=int, default=1)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Stop after frame analysis without running LivePortrait.",
    )
    args = parser.parse_args()

    user_uuid = args.user_uuid or infer_user_uuid(args.object_keys)
    job_id = args.job_id or int(time.time())
    if args.clone_id <= 0 or job_id <= 0:
        raise FaceTrainingWorkerError("clone-id and job-id must be positive.")

    os.environ["FACE_TRAINING_RUN_LIVEPORTRAIT"] = (
        "false" if args.preprocess_only else "true"
    )
    message = FaceTrainingMessage(
        schema_version=1,
        job_type="FACE_PROFILE_BUILD",
        job_id=job_id,
        source="FACE_UPDATE",
        user_uuid=user_uuid,
        clone_id=args.clone_id,
        bucket=args.bucket,
        object_keys=args.object_keys,
    )

    print(
        "[FACE_PREVIEW] local validation started: "
        f"user_uuid={user_uuid} files={len(args.object_keys)}",
        flush=True,
    )
    manifest_path = _preprocess_face_training_message(
        _boto3_client("s3"),
        message,
    )
    print(
        f"[FACE_PREVIEW] completed: manifest={manifest_path}",
        flush=True,
    )


def infer_user_uuid(object_keys: list[str]) -> str:
    inferred = set()
    for object_key in object_keys:
        parts = PurePosixPath(object_key).parts
        try:
            prefix_index = parts.index("face-videos")
            candidate = parts[prefix_index + 1]
        except (ValueError, IndexError) as exc:
            raise FaceTrainingWorkerError(
                "Unable to infer user UUID from object key: " + object_key
            ) from exc
        try:
            UUID(candidate)
        except ValueError as exc:
            raise FaceTrainingWorkerError(
                f"Invalid user UUID in object key: {object_key}"
            ) from exc
        inferred.add(candidate)

    if len(inferred) != 1:
        raise FaceTrainingWorkerError(
            "All preview object keys must belong to the same user."
        )
    return inferred.pop()


if __name__ == "__main__":
    main()
