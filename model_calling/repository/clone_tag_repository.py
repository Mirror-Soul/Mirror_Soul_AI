from typing import Any

from model_calling.clone_tags.generator import CloneTagContext, CloneTagResult
from model_calling.repository.clone_repository import (
    CloneRepositoryError,
    CloneRepositoryNotConfigured,
    _get_db_config,
)


def load_clone_tag_context(*, user_uuid: str) -> CloneTagContext:
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
                    """
                    SELECT
                        c.id AS clone_id,
                        u.id AS user_id,
                        u.uuid AS user_uuid,
                        u.name,
                        u.job,
                        u.job_description,
                        u.self_introduction,
                        mp.mbti
                    FROM users u
                    JOIN clones c ON c.user_id = u.id
                    LEFT JOIN mbti_profile mp ON mp.user_id = u.id
                    WHERE u.uuid = %s
                    LIMIT 1
                    """,
                    (user_uuid,),
                )
                profile = cursor.fetchone()
                if not profile:
                    raise CloneRepositoryError(f"Clone tag context not found: {user_uuid}")

                user_id = profile["user_id"]
                interview_texts = _fetch_interview_texts(cursor, user_id)
                user_talk_texts = _fetch_user_talk_texts(cursor, user_id)
        finally:
            connection.close()
    except CloneRepositoryError:
        raise
    except Exception as exc:
        raise CloneRepositoryError(f"RDS clone tag context lookup failed: {exc}") from exc

    return CloneTagContext(
        clone_id=profile["clone_id"],
        user_uuid=str(profile["user_uuid"]),
        name=profile["name"],
        job=profile["job"],
        job_description=profile["job_description"],
        self_introduction=profile["self_introduction"],
        mbti=profile["mbti"],
        interview_texts=interview_texts,
        user_talk_texts=user_talk_texts,
    )


def save_clone_tags(tag_result: CloneTagResult) -> bool:
    if not tag_result.tags:
        return False

    try:
        import pymysql
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    connection = None
    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM ai_clone_tags
                WHERE clone_id = %s
                  AND source = %s
                """,
                (tag_result.clone_id, tag_result.source),
            )
            for index, tag in enumerate(tag_result.tags, start=1):
                cursor.execute(
                    """
                    INSERT INTO ai_clone_tags (
                        clone_id,
                        tag_text,
                        rank_order,
                        source
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (tag_result.clone_id, tag, index, tag_result.source),
                )
        connection.commit()
        return True
    except Exception as exc:
        if connection:
            connection.rollback()
        if _is_missing_optional_table_error(exc):
            print(
                "[CLONE_TAGS] optional tag table missing; generated tags were not stored",
                flush=True,
            )
            return False
        raise CloneRepositoryError(f"RDS clone tag save failed: {exc}") from exc
    finally:
        if connection:
            connection.close()


def _fetch_interview_texts(cursor: Any, user_id: int) -> list[str]:
    cursor.execute(
        """
        SELECT answer_text
        FROM interview_record
        WHERE user_id = %s
          AND answer_text IS NOT NULL
          AND TRIM(answer_text) <> ''
        ORDER BY id DESC
        LIMIT 10
        """,
        (user_id,),
    )
    return [str(row["answer_text"]) for row in cursor.fetchall()]


def _fetch_user_talk_texts(cursor: Any, user_id: int) -> list[str]:
    cursor.execute(
        """
        SELECT tl.message
        FROM talk_logs tl
        JOIN video_calls vc
          ON vc.id = tl.video_call_id
        WHERE vc.user_id = %s
          AND vc.status = 'COMPLETED'
          AND tl.speaker = 'USER'
          AND tl.message IS NOT NULL
          AND TRIM(tl.message) <> ''
        ORDER BY tl.id DESC
        LIMIT 20
        """,
        (user_id,),
    )
    return [str(row["message"]) for row in cursor.fetchall()]


def _is_missing_optional_table_error(exc: Exception) -> bool:
    args = getattr(exc, "args", ())
    return bool(args and args[0] == 1146)
