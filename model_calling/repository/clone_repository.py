import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class CloneRepositoryError(Exception):
    pass


class CloneRepositoryNotConfigured(CloneRepositoryError):
    pass


class CloneNotFound(CloneRepositoryError):
    pass


@dataclass(frozen=True)
class CloneInfo:
    clone_id: int
    clone_user_uuid: str
    sync_rate: int
    avatar_image_url: str | None
    summary: str | None


@dataclass(frozen=True)
class MemberRuntimeProfile:
    user_uuid: str
    name: str | None
    gender: str | None
    birth_date: date | None
    job: str | None
    job_description: str | None
    self_introduction: str | None
    mbti: str | None


@dataclass(frozen=True)
class ActiveVoiceProfile:
    clone_id: int
    voice_training_job_id: int | None
    elevenlabs_voice_id: str
    status: str
    is_active: bool


@dataclass(frozen=True)
class VoiceTrainingJobFile:
    bucket: str
    object_key: str


def _get_db_config() -> dict[str, Any]:
    config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USERNAME"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
    }

    missing_keys = [
        key
        for key in ("host", "user", "password", "database")
        if not config.get(key)
    ]
    if missing_keys:
        raise CloneRepositoryNotConfigured(
            f"DB config missing: {', '.join(missing_keys)}"
        )

    return config


def find_clone_by_user_uuid(clone_user_uuid: str) -> CloneInfo:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor

    query = """
        SELECT
            c.id AS clone_id,
            u.uuid AS clone_user_uuid,
            c.sync_rate,
            c.avatar_image_url,
            c.summary
        FROM clones c
        JOIN users u ON c.user_id = u.id
        WHERE u.uuid = %s
        LIMIT 1
    """

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (clone_user_uuid,))
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(f"RDS clone lookup failed: {exc}") from exc

    if not row:
        raise CloneNotFound(f"Clone not found: {clone_user_uuid}")

    return CloneInfo(
        clone_id=row["clone_id"],
        clone_user_uuid=str(row["clone_user_uuid"]),
        sync_rate=row["sync_rate"],
        avatar_image_url=row["avatar_image_url"],
        summary=row["summary"],
    )


def find_member_runtime_profile(user_uuid: str) -> MemberRuntimeProfile:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor

    query = """
        SELECT
            u.uuid AS user_uuid,
            u.name,
            u.gender,
            u.birth_date,
            u.job,
            u.job_description,
            u.self_introduction,
            mp.mbti
        FROM users u
        LEFT JOIN mbti_profile mp ON mp.user_id = u.id
        WHERE u.uuid = %s
        LIMIT 1
    """

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_uuid,))
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(f"RDS member profile lookup failed: {exc}") from exc

    if not row:
        raise CloneNotFound(f"Member not found: {user_uuid}")

    return MemberRuntimeProfile(
        user_uuid=str(row["user_uuid"]),
        name=row["name"],
        gender=row["gender"],
        birth_date=row["birth_date"],
        job=row["job"],
        job_description=row["job_description"],
        self_introduction=row["self_introduction"],
        mbti=row["mbti"],
    )


def find_active_voice_profile_by_user_uuid(user_uuid: str) -> ActiveVoiceProfile | None:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor

    query = """
        SELECT
            avp.clone_id,
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
                cursor.execute(query, (user_uuid,))
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(
            f"RDS active voice profile lookup failed: {exc}"
        ) from exc

    if not row:
        return None

    return ActiveVoiceProfile(
        clone_id=row["clone_id"],
        voice_training_job_id=row["voice_training_job_id"],
        elevenlabs_voice_id=row["elevenlabs_voice_id"],
        status=row["status"],
        is_active=bool(row["is_active"]),
    )


def find_voice_training_job_status(job_id: int) -> str | None:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT status FROM voice_training_jobs WHERE id = %s LIMIT 1",
                    (job_id,),
                )
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(
            f"RDS voice training job status lookup failed: {exc}"
        ) from exc

    return row["status"] if row else None


def find_voice_training_job_files(job_id: int) -> list[VoiceTrainingJobFile]:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor

    query = """
        SELECT bucket, object_key
        FROM voice_training_job_files
        WHERE voice_training_job_id = %s
        ORDER BY id
    """

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (job_id,))
                rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(
            f"RDS voice training job files lookup failed: {exc}"
        ) from exc

    return [
        VoiceTrainingJobFile(bucket=row["bucket"], object_key=row["object_key"])
        for row in rows
    ]


def mark_voice_training_job_processing(job_id: int) -> None:
    _update_voice_training_job(
        """
        UPDATE voice_training_jobs
        SET status = 'PROCESSING',
            error_message = NULL,
            started_at = COALESCE(started_at, NOW()),
            finished_at = NULL
        WHERE id = %s
        """,
        (job_id,),
    )


def mark_voice_training_job_failed(job_id: int, error_message: str) -> None:
    _update_voice_training_job(
        """
        UPDATE voice_training_jobs
        SET status = 'FAILED',
            error_message = %s,
            finished_at = NOW()
        WHERE id = %s
        """,
        (_truncate_error(error_message), job_id),
    )


def complete_voice_training_job(
    *,
    job_id: int,
    clone_id: int,
    elevenlabs_voice_id: str,
) -> None:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    config["cursorclass"] = DictCursor
    connection = None

    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ai_voice_profiles
                SET is_active = FALSE
                WHERE clone_id = %s
                """,
                (clone_id,),
            )
            cursor.execute(
                """
                SELECT id
                FROM ai_voice_profiles
                WHERE voice_training_job_id = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (job_id,),
            )
            existing_profile = cursor.fetchone()
            if existing_profile:
                cursor.execute(
                    """
                    UPDATE ai_voice_profiles
                    SET elevenlabs_voice_id = %s,
                        status = 'ACTIVE',
                        is_active = TRUE
                    WHERE id = %s
                    """,
                    (elevenlabs_voice_id, existing_profile["id"]),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO ai_voice_profiles (
                        clone_id,
                        voice_training_job_id,
                        elevenlabs_voice_id,
                        status,
                        is_active
                    )
                    VALUES (%s, %s, %s, 'ACTIVE', TRUE)
                    """,
                    (clone_id, job_id, elevenlabs_voice_id),
                )

            cursor.execute(
                """
                UPDATE voice_training_jobs
                SET status = 'COMPLETED',
                    error_message = NULL,
                    finished_at = NOW()
                WHERE id = %s
                """,
                (job_id,),
            )
        connection.commit()
    except Exception as exc:
        if connection:
            connection.rollback()
        raise CloneRepositoryError(f"RDS voice profile update failed: {exc}") from exc
    finally:
        if connection:
            connection.close()


def _update_voice_training_job(query: str, params: tuple[Any, ...]) -> None:
    try:
        import pymysql
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
            connection.commit()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(f"RDS voice training job update failed: {exc}") from exc


def _truncate_error(error_message: str, limit: int = 2000) -> str:
    normalized = error_message.strip() or "Unknown voice training error"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
