from fastapi import APIRouter

from model_training.clone_training_callback import (
    notify_personality_training_complete,
)
from model_training.schemas import (
    DeleteRagDataResponse,
    MemberProfileRequest,
    MemberProfileResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    TrainingSampleRequest,
    TrainingSampleResponse,
)
from model_training.services import (
    add_member_profile_to_rag,
    add_training_sample_to_rag,
    delete_user_rag_data,
    search_user_memories,
)

router = APIRouter(prefix="/api/v1/training", tags=["model_training"])


@router.post("/samples", response_model=TrainingSampleResponse)
def add_training_sample(request: TrainingSampleRequest):
    result = add_training_sample_to_rag(
        user_id=request.userId,
        ai_profile_id=request.aiProfileId,
        question_id=request.questionId,
        question_category=request.questionCategory,
        question_text=request.questionText,
        transcript=request.transcript,
        mbti=request.mbti,
        description=request.description,
        audio_url=request.audioUrl,
        source_id=request.sourceId,
        clone_id=request.cloneId,
    )

    return TrainingSampleResponse(
        success=True,
        documentId=result["documentId"],
        sampleId=result["sampleId"],
        status=result["status"],
    )


@router.post("/profiles", response_model=MemberProfileResponse)
def add_member_profile(request: MemberProfileRequest):
    interview_samples = [
        sample.model_dump() if hasattr(sample, "model_dump") else sample.dict()
        for sample in request.interviewSamples
    ]

    print(
        "[RAG_PROFILE] processing: "
        f"user_uuid={request.userId} clone_id={request.cloneId} "
        f"samples={len(interview_samples)}",
        flush=True,
    )
    try:
        # The completion callback must only run after the RAG write (upsert and
        # stale cleanup) has fully succeeded.
        result = add_member_profile_to_rag(
            user_id=request.userId,
            ai_profile_id=request.aiProfileId,
            clone_id=request.cloneId,
            name=request.name,
            nickname=request.nickname,
            age=request.age,
            gender=request.gender,
            mbti=request.mbti,
            job=request.job,
            description=request.description,
            interests=request.interests,
            values=request.values,
            interview_topics=request.interviewTopics,
            interview_samples=interview_samples,
            keyword_limit=request.keywordLimit,
        )
        callback_sent = notify_personality_training_complete(
            request.cloneId,
            score_components=result["profileQuality"],
            source_revision=request.sourceRevision,
        )
    except Exception as exc:
        print(
            "[RAG_PROFILE] failed: "
            f"user_uuid={request.userId} clone_id={request.cloneId} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    profile_quality = result["profileQuality"]
    print(
        "[RAG_PROFILE] completed: "
        f"user_uuid={request.userId} clone_id={request.cloneId} "
        f"profile_score={profile_quality.get('profileScore')} "
        f"data_reliability={profile_quality.get('dataReliabilityScore')} "
        f"penalty={profile_quality.get('penaltyScore', 0.0)} "
        f"revision={request.sourceRevision or 'none'} "
        f"documents={result.get('documentCount')} "
        f"removed={result.get('removedDocumentCount')} "
        f"callback_sent={callback_sent}",
        flush=True,
    )

    return MemberProfileResponse(
        success=True,
        documentId=result["documentId"],
        status=result["status"],
        keywords=result["keywords"],
        profileSummary=result["profileSummary"],
    )


@router.post("/search", response_model=SearchMemoryResponse)
def search_training_memories(request: SearchMemoryRequest):
    memories = search_user_memories(
        user_id=request.userId,
        query=request.query,
        top_k=request.topK,
    )

    return SearchMemoryResponse(
        success=True,
        memories=memories,
    )


@router.delete("/users/{user_id}/rag-data", response_model=DeleteRagDataResponse)
def delete_training_rag_data(user_id: str):
    result = delete_user_rag_data(user_id)

    return DeleteRagDataResponse(
        success=result["success"],
        userId=result["userId"],
        deleted=result["deleted"],
    )
