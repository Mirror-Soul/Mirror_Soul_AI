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

    AI_SERVER_HOST: str = os.getenv("AI_SERVER_HOST", "0.0.0.0")
    AI_SERVER_PORT: int = int(os.getenv("AI_SERVER_PORT", "8000"))


settings = Settings()