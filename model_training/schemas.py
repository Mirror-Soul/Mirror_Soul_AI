from pydantic import BaseModel, Field


class TrainingSampleRequest(BaseModel):
    userId: str = Field(..., description="백엔드 회원 ID")
    aiProfileId: str = Field(..., description="AI 프로필 ID")
    cloneId: int | None = Field(None, gt=0, description="선택: 클론 ID")
    sourceId: str | None = Field(
        None,
        description="선택: 인터뷰 답변의 안정적인 원본 ID. 없으면 questionId로 생성",
    )
    mbti: str | None = Field(None, description="회원 MBTI (호환용, 인터뷰 문서에는 저장하지 않음)")
    description: str | None = Field(None, description="회원 자기소개 (호환용, 인터뷰 문서에는 저장하지 않음)")
    questionId: int = Field(..., description="인터뷰 질문 ID")
    questionCategory: str = Field(..., description="인터뷰 질문 카테고리")
    questionText: str = Field(..., description="인터뷰 질문 내용")
    transcript: str = Field(..., description="사용자 답변 STT 텍스트")
    audioUrl: str | None = Field(None, description="백엔드가 저장한 음성 파일 URL")


class TrainingSampleResponse(BaseModel):
    success: bool
    documentId: str
    sampleId: str = Field(..., description="문서 sourceId (구버전 호환 필드명)")
    status: str


class InterviewKeywordSource(BaseModel):
    sourceId: str | None = Field(
        None,
        description="선택: 인터뷰 답변의 안정적인 원본 ID. 없으면 questionId로 생성",
    )
    questionId: int | None = Field(None, description="인터뷰 질문 ID")
    questionCategory: str | None = Field(None, description="인터뷰 질문 카테고리")
    questionText: str | None = Field(None, description="인터뷰 질문 내용")
    transcript: str = Field(..., description="사용자 답변 STT 텍스트")


class MemberProfileRequest(BaseModel):
    userId: str = Field(..., description="백엔드 회원 ID")
    cloneId: int = Field(..., gt=0, description="완료 처리할 클론 ID")
    aiProfileId: str | None = Field(None, description="AI 프로필 ID")
    name: str | None = Field(None, description="선택: 회원 이름")
    nickname: str | None = Field(None, description="선택: 회원 닉네임")
    age: int | None = Field(None, ge=0, le=130, description="회원 나이")
    gender: str | None = Field(None, description="회원 성별")
    mbti: str | None = Field(None, description="회원 MBTI")
    job: str | None = Field(None, description="선택: 회원 직업")
    description: str | None = Field(None, description="회원 자기소개")
    interests: list[str] = Field(default_factory=list, description="백엔드가 전달한 관심사")
    values: list[str] = Field(default_factory=list, description="선택: 회원 가치관")
    interviewTopics: list[str] = Field(
        default_factory=list,
        description="백엔드가 전달한 인터뷰 주요 주제",
    )
    interviewSamples: list[InterviewKeywordSource] = Field(
        default_factory=list,
        description="키워드 추출에 사용할 인터뷰 답변 샘플",
    )
    keywordLimit: int = Field(12, ge=1, le=30, description="저장할 최대 키워드 수")
    sourceRevision: int | None = Field(
        None,
        gt=0,
        description=(
            "선택: 백엔드 RAG 프로필 작업 revision. 완료 콜백에 그대로 돌려줘서 "
            "다시 학습한 프로필 점수가 이전 점수를 대체하게 함"
        ),
    )


class MemberProfileResponse(BaseModel):
    success: bool
    documentId: str
    status: str
    keywords: list[str]
    profileSummary: str


class SearchMemoryRequest(BaseModel):
    userId: str = Field(..., description="검색 대상 회원 ID")
    query: str = Field(..., description="검색 질문")
    topK: int = Field(5, description="검색 결과 개수")


class MemoryItem(BaseModel):
    documentId: str
    text: str
    metadata: dict
    distance: float | None = None


class SearchMemoryResponse(BaseModel):
    success: bool
    memories: list[MemoryItem]


class DeleteRagDataResponse(BaseModel):
    success: bool
    userId: str
    deleted: str
