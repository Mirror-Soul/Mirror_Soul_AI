import json
from dataclasses import dataclass
from uuid import UUID


class FaceTrainingMessageError(ValueError):
    pass


@dataclass(frozen=True)
class FaceTrainingMessage:
    schema_version: int
    job_type: str
    job_id: int
    source: str
    user_uuid: str
    clone_id: int
    bucket: str
    object_keys: list[str]


def parse_face_training_message(message_body: str) -> FaceTrainingMessage:
    try:
        data = json.loads(message_body)
    except json.JSONDecodeError as exc:
        raise FaceTrainingMessageError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise FaceTrainingMessageError("Message body must be a JSON object.")

    missing_fields = [
        field
        for field in (
            "schemaVersion",
            "jobType",
            "jobId",
            "source",
            "userUuid",
            "cloneId",
            "bucket",
            "objectKeys",
        )
        if data.get(field) in (None, "")
    ]
    if missing_fields:
        raise FaceTrainingMessageError(
            f"Missing required field(s): {', '.join(missing_fields)}"
        )

    try:
        schema_version = int(data["schemaVersion"])
        job_id = int(data["jobId"])
        clone_id = int(data["cloneId"])
    except (TypeError, ValueError) as exc:
        raise FaceTrainingMessageError(
            "schemaVersion, jobId, and cloneId must be integers."
        ) from exc

    if schema_version != 1:
        raise FaceTrainingMessageError(
            f"Unsupported schemaVersion: {schema_version}"
        )
    if str(data["jobType"]) != "FACE_PROFILE_BUILD":
        raise FaceTrainingMessageError(
            f"Unsupported jobType: {data['jobType']}"
        )
    if str(data["source"]) not in {"ONBOARDING_FACE", "FACE_UPDATE"}:
        raise FaceTrainingMessageError(f"Unsupported source: {data['source']}")
    if job_id <= 0 or clone_id <= 0:
        raise FaceTrainingMessageError("jobId and cloneId must be positive integers.")

    user_uuid = str(data["userUuid"])
    try:
        UUID(user_uuid)
    except ValueError as exc:
        raise FaceTrainingMessageError("userUuid must be a valid UUID.") from exc

    object_keys = data["objectKeys"]
    if not isinstance(object_keys, list) or not object_keys:
        raise FaceTrainingMessageError("objectKeys must be a non-empty list.")

    normalized_object_keys = []
    for object_key in object_keys:
        normalized = str(object_key).strip().replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise FaceTrainingMessageError(f"Invalid objectKey: {object_key}")
        normalized_object_keys.append(normalized)

    return FaceTrainingMessage(
        schema_version=schema_version,
        job_type="FACE_PROFILE_BUILD",
        job_id=job_id,
        source=str(data["source"]),
        user_uuid=user_uuid,
        clone_id=clone_id,
        bucket=str(data["bucket"]),
        object_keys=normalized_object_keys,
    )

