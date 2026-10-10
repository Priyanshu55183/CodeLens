"""
Configuration module for CodeLens.

Design decision: All settings come from environment variables (via .env file).
This keeps secrets out of code and makes the app configurable without code changes.
We use pydantic's BaseSettings for automatic env var parsing and validation.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Interview note: pydantic-settings automatically reads from .env files
    and validates types. If a required field is missing, you get a clear
    error at startup rather than a crash later.
    """

    # ── LLM Provider ──────────────────────────────────────────
    llm_provider: str = "openai"          # "openai" | "anthropic" | "gemini"
    llm_model: str = "gpt-4o-mini"        # Model for answering questions
    embedding_model: str = "text-embedding-3-small"  # Model for embeddings

    # ── API Keys ──────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # ── Optional ──────────────────────────────────────────────
    github_token: str = ""                 # For private repos

    # ── Storage ───────────────────────────────────────────────
    data_dir: Path = Path("./data")        # Root for all persistent data

    # ── Chunker ───────────────────────────────────────────────
    chunk_size_lines: int = 80             # Lines per chunk
    chunk_overlap_lines: int = 10          # Overlap between chunks

    # ── Retrieval ─────────────────────────────────────────────
    top_k: int = 10                        # Number of chunks to retrieve

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def repos_dir(self) -> Path:
        """Directory where cloned repos are stored."""
        return self.data_dir / "repos"

    @property
    def chroma_dir(self) -> Path:
        """Directory where ChromaDB persists data."""
        return self.data_dir / "chroma"


def get_settings() -> Settings:
    """Create and return a Settings instance.
    
    Not cached as a singleton so tests can override env vars freely.
    """
    return Settings()
