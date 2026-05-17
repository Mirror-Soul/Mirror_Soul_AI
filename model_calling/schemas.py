from pydantic import BaseModel, Field
from typing import List, Optional

class UserStyle(BaseModel):
    frequent_words: List[str] = Field(default_factory=list)
    sentence_endings: str = ""
    fillers: List[str] = Field(default_factory=list)
    sentence_style: str = ""

class PersonalityProfile(BaseModel):
    openness: float = Field(..., ge=0, le=100)
    conscientiousness: float = Field(..., ge=0, le=100)
    extraversion: float = Field(..., ge=0, le=100)
    agreeableness: float = Field(..., ge=0, le=100)
    neuroticism: float = Field(..., ge=0, le=100)
    summary: str

class SpeechProfile(BaseModel):
    user_id: str
    voice_id: Optional[str] = None
    speech_speed: float = Field(..., ge=0, le=100)
    avg_pitch: float = Field(..., ge=0, le=100)
    honorific_ratio: float = Field(..., ge=0, le=100)
    summary: str
    user_style: Optional[UserStyle] = None