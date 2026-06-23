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


def _get_db_config() -> dict[str, Any]:
    config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USERNAME"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "cursorclass": None,
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
