import os
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CloneTagContext:
    clone_id: int
    user_uuid: str
    name: str | None
    job: str | None
    job_description: str | None
    self_introduction: str | None
    mbti: str | None
    interview_texts: list[str]
    user_talk_texts: list[str]


@dataclass(frozen=True)
class CloneTagResult:
    clone_id: int
    tags: list[str]
    source: str = "AI_PERSONALITY_SCAN"


KEYWORD_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "\uc608\uc220\uc801 \uac10\uc218\uc131",
        (
            "\uc608\uc220",
            "\uc74c\uc545",
            "\ub178\ub798",
            "\uc804\uc2dc",
            "\uadf8\ub9bc",
            "\ub514\uc790\uc778",
            "\uc0ac\uc9c4",
            "\uc601\ud654",
            "\uae00",
            "\ucc45",
        ),
    ),
    (
        "\uc131\uc7a5 \uc9c0\ud5a5\uc801",
        (
            "\uc131\uc7a5",
            "\uc790\uae30\uacc4\ubc1c",
            "\ubc30\uc6b0",
            "\ubc1c\uc804",
            "\ubaa9\ud45c",
            "\ub3c4\uc804",
            "\ub178\ub825",
            "\uacf5\ubd80",
        ),
    ),
    (
        "\uac74\uac15\ud55c \ud65c\ub3d9\uac00",
        (
            "\uc6b4\ub3d9",
            "\ud5ec\uc2a4",
            "\ub7ec\ub2dd",
            "\ub2ec\ub9ac\uae30",
            "\uc694\uac00",
            "\ub4f1\uc0b0",
            "\ucd95\uad6c",
            "\ub18d\uad6c",
            "\uac74\uac15",
            "\uccb4\ub825",
        ),
    ),
    (
        "\uc0ac\ub78c\uc744 \uc88b\uc544\ud558\ub294",
        (
            "\uc0ac\ub78c",
            "\uce5c\uad6c",
            "\ud568\uaed8",
            "\ub300\ud654",
            "\ud30c\ud2f0",
            "\ubaa8\uc784",
            "\uc18c\ud1b5",
            "\uc5d0\ub108\uc9c0",
        ),
    ),
    (
        "\ucc28\ubd84\ud55c \ub9d0\ud22c",
        (
            "\ucc28\ubd84",
            "\uc870\uc6a9",
            "\ud63c\uc790",
            "\ud734\uc2dd",
            "\uc0b0\ucc45",
            "\uc0dd\uac01",
            "\uc815\ub9ac",
        ),
    ),
    (
        "\uc0c8\ub85c\uc6b4 \uacbd\ud5d8 \ucd94\uad6c",
        (
            "\uc5ec\ud589",
            "\uc0c8\ub85c\uc6b4",
            "\uacbd\ud5d8",
            "\ud0d0\ud5d8",
            "\uc2dc\ub3c4",
            "\ubaa8\ud5d8",
        ),
    ),
    (
        "\ubab0\uc785\ud558\ub294 \uc2e4\ubb34\ud615",
        (
            "\uc77c",
            "\uc9c1\uc7a5",
            "\ud504\ub85c\uc81d\ud2b8",
            "\uac1c\ubc1c",
            "\uae30\ud68d",
            "\ubd84\uc11d",
            "\uc5c5\ubb34",
            "\ubb38\uc81c",
        ),
    ),
    (
        "\uacf5\uac10\uc774 \uae4a\uc740",
        (
            "\uacf5\uac10",
            "\uac10\uc815",
            "\ub9c8\uc74c",
            "\ubc30\ub824",
            "\uc704\ub85c",
            "\ub530\ub73b",
            "\uc774\ud574",
        ),
    ),
    (
        "\ub17c\ub9ac\uc801\uc778",
        (
            "\ub17c\ub9ac",
            "\ubd84\uc11d",
            "\uc774\uc131",
            "\uacc4\ud68d",
            "\ud310\ub2e8",
            "\ud6a8\uc728",
            "\uc815\ud655",
        ),
    ),
)


def generate_clone_tags(context: CloneTagContext) -> CloneTagResult:
    max_tags = _env_int("CLONE_TAG_MAX_COUNT", 4)
    candidates: list[str] = []

    candidates.extend(_tags_from_keywords(_combined_text(context)))
    candidates.extend(_tags_from_mbti(context.mbti))
    candidates.extend(_fallback_tags(context))

    return CloneTagResult(
        clone_id=context.clone_id,
        tags=_dedupe_tags(candidates, max_tags=max_tags),
    )


def _tags_from_mbti(mbti: str | None) -> list[str]:
    normalized = (mbti or "").upper().strip()
    if len(normalized) < 4:
        return []

    tags: list[str] = []
    tags.append(
        "\uc0ac\ub78c\uc5d0\uac8c \uc5d0\ub108\uc9c0"
        if normalized[0] == "E"
        else "\ucc28\ubd84\ud55c \uad00\ucc30\uc790"
    )
    tags.append(
        "\uc0c1\uc0c1\ub825\uc774 \ud48d\ubd80\ud55c"
        if normalized[1] == "N"
        else "\ud604\uc2e4 \uac10\uac01 \uc88b\uc740"
    )
    tags.append(
        "\uacf5\uac10\uc774 \uae4a\uc740"
        if normalized[2] == "F"
        else "\ub17c\ub9ac\uc801\uc778"
    )
    tags.append(
        "\uc720\uc5f0\ud55c \ud0d0\uc0c9\uac00"
        if normalized[3] == "P"
        else "\uacc4\ud68d\uc801\uc778 \uc2e4\ud589\uac00"
    )
    return tags


def _tags_from_keywords(text: str) -> list[str]:
    compact_text = re.sub(r"\s+", "", text.lower())
    scored_tags: list[tuple[str, int]] = []
    for tag, keywords in KEYWORD_TAGS:
        score = sum(compact_text.count(keyword.lower()) for keyword in keywords)
        if score > 0:
            scored_tags.append((tag, score))

    scored_tags.sort(key=lambda item: item[1], reverse=True)
    return [tag for tag, _ in scored_tags]


def _fallback_tags(context: CloneTagContext) -> list[str]:
    tags: list[str] = []
    if _has_text(context.job) or _has_text(context.job_description):
        tags.append("\uc790\uae30 \ubd84\uc57c\uac00 \ub69c\ub837\ud55c")
    if _has_text(context.self_introduction):
        tags.append("\uc790\uae30\ud45c\ud604\uc774 \ubd84\uba85\ud55c")
    tags.extend(
        [
            "\uc0ac\uace0\uac00 \uae4a\uc740",
            "\uc131\uc7a5 \uc9c0\ud5a5\uc801",
            "\ucc28\ubd84\ud55c \ub9d0\ud22c",
        ]
    )
    return tags


def _combined_text(context: CloneTagContext) -> str:
    parts: list[str] = [
        context.name or "",
        context.job or "",
        context.job_description or "",
        context.self_introduction or "",
        context.mbti or "",
    ]
    parts.extend(_clean_texts(context.interview_texts))
    parts.extend(_clean_texts(context.user_talk_texts))
    return " ".join(parts)


def _clean_texts(values: Iterable[str]) -> list[str]:
    return [value.strip() for value in values if _has_text(value)]


def _dedupe_tags(tags: Iterable[str], *, max_tags: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = _normalize_tag(tag)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= max(1, max_tags):
            break
    return result


def _normalize_tag(tag: str) -> str:
    return tag.strip().removeprefix("#").strip()


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default
