import re
from uuid import uuid4


KEYWORD_STOPWORDS = {
    "그리고",
    "그래서",
    "하지만",
    "저는",
    "제가",
    "나는",
    "나를",
    "우리",
    "정말",
    "너무",
    "많이",
    "조금",
    "자주",
    "그냥",
    "이런",
    "저런",
    "가장",
    "것을",
    "것이",
    "것도",
    "때문",
    "대한",
    "대해",
    "이야기",
    "중요하게",
    "생각하",
    "생각하는",
    "무엇인가요",
    "좋아하고",
    "해요",
    "합니다",
    "했다",
    "하고",
    "하는",
    "하면",
    "있습니다",
    "있어요",
    "없어요",
    "생각합니다",
    "생각해요",
}


def create_sample_id() -> str:
    """Create a unique identifier for a single training sample."""
    return f"sample_{uuid4().hex}"


def create_document_id(sample_id: str) -> str:
    """Create the vector-store document id derived from a sample id."""
    return f"training_{sample_id}"


def create_member_profile_document_id(user_id: str, ai_profile_id: str | None = None) -> str:
    """Create a stable document id for a user's processed profile summary."""
    profile_key = ai_profile_id or "default"
    safe_user_id = re.sub(r"[^0-9A-Za-z_.-]", "_", user_id)
    safe_profile_key = re.sub(r"[^0-9A-Za-z_.-]", "_", profile_key)
    return f"member_profile_{safe_user_id}_{safe_profile_key}"


def normalize_keyword(keyword: str) -> str:
    normalized = re.sub(r"\s+", " ", keyword.strip())
    normalized = normalized.strip(".,!?;:()[]{}\"'")

    if re.fullmatch(r"[가-힣]{3,}", normalized):
        for suffix in ("으로", "에서", "에게", "한테", "보다", "처럼", "까지", "부터"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                return normalized[: -len(suffix)]

        for suffix in ("은", "는", "이", "가", "을", "를", "과", "와", "에", "로", "도", "만", "의"):
            if normalized.endswith(suffix) and len(normalized) > 2:
                return normalized[:-1]

    return normalized


def extract_keywords_from_texts(
    texts: list[str],
    *,
    seed_keywords: list[str] | None = None,
    limit: int = 12,
) -> list[str]:
    """Extract compact profile keywords without storing long raw member text."""
    ordered_keywords: list[str] = []
    seen: set[str] = set()

    def add_keyword(keyword: str) -> None:
        normalized = normalize_keyword(keyword)
        lowered = normalized.lower()
        if (
            not normalized
            or lowered in seen
            or normalized in KEYWORD_STOPWORDS
            or len(normalized) < 2
        ):
            return

        seen.add(lowered)
        ordered_keywords.append(normalized)

    for keyword in seed_keywords or []:
        add_keyword(keyword)

    token_counts: dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[가-힣A-Za-z0-9+#_.-]{2,}", text):
            normalized = normalize_keyword(token)
            if not normalized or normalized in KEYWORD_STOPWORDS:
                continue
            token_counts[normalized] = token_counts.get(normalized, 0) + 1

    for keyword, _ in sorted(
        token_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        add_keyword(keyword)
        if len(ordered_keywords) >= limit:
            break

    return ordered_keywords[:limit]


def build_member_profile_summary_text(
    *,
    age: int | None,
    gender: str | None,
    mbti: str | None,
    keywords: list[str],
) -> str:
    """Build a compact RAG document for important member profile signals."""
    sections = [
        "[회원 핵심 프로필]",
        f"나이: {age if age is not None else '미입력'}",
        f"성별: {gender.strip() if gender else '미입력'}",
        f"MBTI: {mbti.strip().upper() if mbti else '미입력'}",
        "",
        "[핵심 키워드]",
    ]

    if keywords:
        sections.extend(f"- {keyword}" for keyword in keywords)
    else:
        sections.append("- 미입력")

    return "\n".join(sections)


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
