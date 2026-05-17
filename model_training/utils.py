from uuid import uuid4


def create_sample_id() -> str:
    """Create a unique identifier for a single training sample."""
    return f"sample_{uuid4().hex}"


def create_document_id(sample_id: str) -> str:
    """Create the vector-store document id derived from a sample id."""
    return f"training_{sample_id}"


def build_training_text(
    *,
    mbti: str | None,
    description: str | None,
    question_category: str,
    question_text: str,
    transcript: str,
) -> str:
    """Build the text payload embedded into the RAG vector store.

    The format keeps user profile hints, the interview question, and the
    user's answer together so semantic search can retrieve the sample with
    enough context for downstream persona generation or chat calls.
    """
    sections = [
        "[회원 프로필]",
        f"MBTI: {mbti.strip() if mbti else '미입력'}",
        f"자기소개: {description.strip() if description else '미입력'}",
        "",
        "[인터뷰 질문]",
        f"카테고리: {question_category.strip()}",
        f"질문: {question_text.strip()}",
        "",
        "[사용자 답변]",
        transcript.strip(),
    ]

    return "\n".join(sections)