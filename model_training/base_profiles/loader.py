import json
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_PROFILE_DIR = Path(__file__).resolve().parent
VALID_MBTI_TYPES = {
    "ISTJ",
    "ISFJ",
    "INFJ",
    "INTJ",
    "ISTP",
    "ISFP",
    "INFP",
    "INTP",
    "ESTP",
    "ESFP",
    "ENFP",
    "ENTP",
    "ESTJ",
    "ESFJ",
    "ENFJ",
    "ENTJ",
}


@lru_cache(maxsize=len(VALID_MBTI_TYPES))
def get_mbti_base_profile(mbti: str | None) -> dict[str, Any] | None:
    if not mbti:
        return None

    normalized_mbti = mbti.strip().upper()
    if normalized_mbti not in VALID_MBTI_TYPES:
        return None

    profile_path = BASE_PROFILE_DIR / f"{normalized_mbti}.json"
    with profile_path.open("r", encoding="utf-8") as profile_file:
        return json.load(profile_file)
