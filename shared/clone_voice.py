import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID


class CloneVoiceError(RuntimeError):
    pass


class CloneVoiceNotConfigured(CloneVoiceError):
    pass


class CloneVoiceNotFound(CloneVoiceError):
    pass


class CloneVoiceMismatch(CloneVoiceError):
    pass


@dataclass(frozen=True)
class ActiveCloneVoice:
    clone_id: int
    user_uuid: str
    voice_training_job_id: int | None
    elevenlabs_voice_id: str
    status: str
    is_active: bool


def find_active_clone_voice(
    user_uuid: str,
    *,
    expected_clone_id: int | None = None,
) -> ActiveCloneVoice:
    normalized_user_uuid = _validate_user_uuid(user_uuid)
    if expected_clone_id is not None and expected_clone_id <= 0:
        raise ValueError("expected_clone_id must be positive")

    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneVoiceNotConfigured(
            "PyMySQL is required to resolve member voice profiles."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor
    query = """
        SELECT
            avp.clone_id,
            u.uuid AS user_uuid,
            avp.voice_training_job_id,
            avp.elevenlabs_voice_id,
            avp.status,
            avp.is_active
        FROM ai_voice_profiles avp
        JOIN clones c ON c.id = avp.clone_id
        JOIN users u ON u.id = c.user_id
        WHERE u.uuid = %s
          AND avp.status = 'ACTIVE'
          AND avp.is_active = TRUE
        ORDER BY avp.updated_at DESC
        LIMIT 1
    """

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (normalized_user_uuid,))
                row = cursor.fetchone()
        finally:
            connection.close()
    except CloneVoiceError:
        raise
    except Exception as exc:
        raise CloneVoiceError(f"RDS active voice lookup failed: {exc}") from exc

    if not row:
        raise CloneVoiceNotFound(
            f"Active voice profile not found for user: {normalized_user_uuid}"
        )

    clone_id = int(row["clone_id"])
    if expected_clone_id is not None and clone_id != expected_clone_id:
        raise CloneVoiceMismatch(
            "Face and voice profiles belong to different clones: "
            f"expected={expected_clone_id} actual={clone_id}"
        )

    voice_id = str(row.get("elevenlabs_voice_id") or "").strip()
    if not voice_id:
        raise CloneVoiceNotFound(
            f"ElevenLabs voice ID is missing for clone: {clone_id}"
        )

    return ActiveCloneVoice(
        clone_id=clone_id,
        user_uuid=str(row["user_uuid"]),
        voice_training_job_id=row.get("voice_training_job_id"),
        elevenlabs_voice_id=voice_id,
        status=str(row["status"]),
        is_active=bool(row["is_active"]),
    )


def _get_db_config() -> dict[str, Any]:
    config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USERNAME"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
    }
    missing = [
        name
        for name in ("host", "user", "password", "database")
        if not config.get(name)
    ]
    if missing:
        raise CloneVoiceNotConfigured(
            f"DB config missing: {', '.join(missing)}"
        )
    return config


def _validate_user_uuid(user_uuid: str) -> str:
    normalized = str(user_uuid).strip()
    try:
        UUID(normalized)
    except ValueError as exc:
        raise ValueError("user_uuid must be a valid UUID") from exc
    return normalized
