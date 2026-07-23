from typing import Any

from model_calling.clone_similarity.scorer import (
    CloneSimilarityScore,
    CloneSimilaritySnapshot,
)
from model_calling.repository.clone_repository import (
    CloneRepositoryError,
    CloneRepositoryNotConfigured,
    _get_db_config,
)


def load_clone_similarity_snapshot(
    *,
    user_uuid: str,
    voice_training_job_id: int | None = None,
) -> CloneSimilaritySnapshot:
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
            u.uuid AS user_uuid,
            u.name,
            u.gender,
            u.birth_date,
            u.job,
            u.job_description,
            u.self_introduction,
            mp.mbti,
            avp.id AS voice_profile_id,
            avp.voice_training_job_id,
            avp.elevenlabs_voice_id,
            vtj.status AS voice_training_status,
            (
                SELECT COUNT(*)
                FROM voice_training_job_files vtjf
                JOIN voice_training_jobs vtj2
                  ON vtj2.id = vtjf.voice_training_job_id
                WHERE vtj2.user_id = u.id
                  AND vtj2.status = 'COMPLETED'
            ) AS voice_training_audio_count,
            (
                SELECT COUNT(*)
                FROM interview_record ir
                WHERE ir.user_id = u.id
            ) AS interview_answer_count,
            (
                SELECT COUNT(*)
                FROM interview_record ir
                WHERE ir.user_id = u.id
                  AND ir.answer_text IS NOT NULL
                  AND TRIM(ir.answer_text) <> ''
            ) AS interview_text_count,
            (
                SELECT COUNT(*)
                FROM interview_record ir
                WHERE ir.user_id = u.id
                  AND ir.answer_audio_object_key IS NOT NULL
                  AND TRIM(ir.answer_audio_object_key) <> ''
            ) AS interview_audio_count
        FROM users u
        JOIN clones c ON c.user_id = u.id
        LEFT JOIN mbti_profile mp ON mp.user_id = u.id
        LEFT JOIN ai_voice_profiles avp
          ON avp.clone_id = c.id
         AND avp.status = 'ACTIVE'
         AND avp.is_active = TRUE
        LEFT JOIN voice_training_jobs vtj
          ON vtj.id = COALESCE(%s, avp.voice_training_job_id)
        WHERE u.uuid = %s
        ORDER BY avp.updated_at DESC
        LIMIT 1
    """

    try:
        connection = pymysql.connect(**config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, (voice_training_job_id, user_uuid))
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise CloneRepositoryError(
            f"RDS clone similarity snapshot lookup failed: {exc}"
        ) from exc

    if not row:
        raise CloneRepositoryError(f"Clone similarity snapshot not found: {user_uuid}")

    return CloneSimilaritySnapshot(
        clone_id=row["clone_id"],
        user_uuid=str(row["user_uuid"]),
        name=row["name"],
        gender=row["gender"],
        birth_date=row["birth_date"],
        job=row["job"],
        job_description=row["job_description"],
        self_introduction=row["self_introduction"],
        mbti=row["mbti"],
        voice_profile_id=row["voice_profile_id"],
        voice_training_job_id=row["voice_training_job_id"] or voice_training_job_id,
        elevenlabs_voice_id=row["elevenlabs_voice_id"],
        voice_training_status=row["voice_training_status"],
        voice_training_audio_count=int(row["voice_training_audio_count"] or 0),
        interview_answer_count=int(row["interview_answer_count"] or 0),
        interview_text_count=int(row["interview_text_count"] or 0),
        interview_audio_count=int(row["interview_audio_count"] or 0),
    )


def save_clone_similarity_score(score: CloneSimilarityScore) -> bool:
    try:
        import pymysql
    except ImportError as exc:
        raise CloneRepositoryNotConfigured(
            "PyMySQL is not installed. Add PyMySQL to requirements.txt and install it."
        ) from exc

    config = _get_db_config()
    connection = None
    detail_saved = False

    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE clones
                SET sync_rate = %s
                WHERE id = %s
                """,
                (round(score.total_score), score.clone_id),
            )
            try:
                _insert_similarity_detail(cursor, score)
                detail_saved = True
            except Exception as exc:
                if not _is_missing_optional_table_error(exc):
                    raise
                print(
                    "[CLONE_SIMILARITY] optional detail table missing; "
                    "stored total score in clones.sync_rate only",
                    flush=True,
                )
        connection.commit()
        return detail_saved
    except Exception as exc:
        if connection:
            connection.rollback()
        raise CloneRepositoryError(f"RDS clone similarity save failed: {exc}") from exc
    finally:
        if connection:
            connection.close()


def _insert_similarity_detail(cursor: Any, score: CloneSimilarityScore) -> None:
    cursor.execute(
        """
        INSERT INTO ai_clone_similarity_scores (
            clone_id,
            voice_profile_id,
            voice_training_job_id,
            voice_score,
            interview_score,
            profile_score,
            total_score,
            explanation,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'COMPLETED')
        """,
        (
            score.clone_id,
            score.voice_profile_id,
            score.voice_training_job_id,
            score.voice_score,
            score.interview_score,
            score.profile_score,
            score.total_score,
            score.explanation,
        ),
    )


def _is_missing_optional_table_error(exc: Exception) -> bool:
    args = getattr(exc, "args", ())
    return bool(args and args[0] == 1146)
