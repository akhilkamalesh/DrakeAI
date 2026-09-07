import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Root workspace directory
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = WORKSPACE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)


def _get_gemini_model() -> str:
    val = (os.getenv("GEMINI_MODEL") or "gemini-flash-latest").strip()
    if not val or "2.5" in val:
        return "gemini-flash-latest"
    return val


class Settings(BaseModel):
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/drake")
    DB_POOL_MIN: int = int(os.getenv("DB_POOL_MIN", "1"))
    DB_POOL_MAX: int = int(os.getenv("DB_POOL_MAX", "10"))

    # Backend API Server
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # Gemini LLM Provider Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    GEMINI_MODEL: str = _get_gemini_model()

    # Local Ollama Configuration (Legacy / Fallback)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gpt-oss:latest")
    
    # Optional Cloud API keys (fallback)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    TARGET_EMBEDDING_DIM: int = int(os.getenv("TARGET_EMBEDDING_DIM", "1024"))


settings = Settings()
