import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    RAG_DB_PATH: str = os.getenv("RAG_DB_PATH", "./rag_store/chroma")
    RAG_COLLECTION_NAME: str = os.getenv(
        "RAG_COLLECTION_NAME",
        "mirror_soul_memories",
    )
    RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.75"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))
    RAG_CONTEXT_MAX_CHARS: int = int(
        os.getenv("RAG_CONTEXT_MAX_CHARS", "3000")
    )
    REALTIME_HISTORY_MAX_TURNS: int = int(
        os.getenv("REALTIME_HISTORY_MAX_TURNS", "8")
    )
    REALTIME_RAG_QUERY_HISTORY_TURNS: int = int(
        os.getenv("REALTIME_RAG_QUERY_HISTORY_TURNS", "2")
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "200"))
    LLM_RESPONSE_MAX_CHARS: int = int(
        os.getenv("LLM_RESPONSE_MAX_CHARS", "240")
    )
    LLM_RESPONSE_MAX_SENTENCES: int = int(
        os.getenv("LLM_RESPONSE_MAX_SENTENCES", "3")
    )
    LLM_PERSONALITY_SIGNAL_THRESHOLD: float = float(
        os.getenv("LLM_PERSONALITY_SIGNAL_THRESHOLD", "12")
    )
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "none")
    STT_MODEL: str = os.getenv("STT_MODEL", "whisper-1")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "ko")
    REALTIME_STT_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_STT_TIMEOUT_SECONDS", "30")
    )
    REALTIME_CONTEXT_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_CONTEXT_TIMEOUT_SECONDS", "15")
    )
    REALTIME_RAG_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_RAG_TIMEOUT_SECONDS", "10")
    )
    REALTIME_LLM_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_LLM_TIMEOUT_SECONDS", "30")
    )
    REALTIME_TTS_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_TTS_TIMEOUT_SECONDS", "45")
    )
    REALTIME_VIDEO_TIMEOUT_SECONDS: float = float(
        os.getenv("REALTIME_VIDEO_TIMEOUT_SECONDS", "180")
    )
    REALTIME_UTTERANCE_MAX_QUEUE_SECONDS: float = float(
        os.getenv("REALTIME_UTTERANCE_MAX_QUEUE_SECONDS", "20")
    )

    AI_SERVER_HOST: str = os.getenv("AI_SERVER_HOST", "0.0.0.0")
    AI_SERVER_PORT: int = int(os.getenv("AI_SERVER_PORT", "8000"))


settings = Settings()
