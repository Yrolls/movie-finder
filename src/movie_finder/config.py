"""All runtime settings, read from the environment (and an optional .env file).

Nothing else in the project reads os.environ directly -- import `get_settings()` instead.
Secrets (API key / base URL) are never given defaults, so a missing key fails loudly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = PROJECT_ROOT / "eval"


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from e


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from e


@dataclass(frozen=True)
class Settings:
    # LLM proxy
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str
    # embeddings
    embed_model: str
    # retrieval
    chunk_words: int
    chunk_overlap: int
    top_k: int
    min_score: float
    retrieval_mode: str
    # agent
    agent_max_steps: int

    def require_llm(self) -> None:
        """Call before any LLM request; gives a clear message instead of an auth error."""
        missing = [n for n, v in (("OPENAI_API_KEY", self.openai_api_key),
                                  ("OPENAI_BASE_URL", self.openai_base_url)) if not v]
        if missing:
            raise ConfigError(
                f"Missing {', '.join(missing)}. Copy .env.example to .env and fill them in."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL", "gemma-4-E4B-it"),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        chunk_words=_int("CHUNK_WORDS", 200),
        chunk_overlap=_int("CHUNK_OVERLAP", 50),
        top_k=_int("TOP_K", 5),
        min_score=_float("MIN_SCORE", 0.82),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        agent_max_steps=_int("AGENT_MAX_STEPS", 5),
    )
    if s.chunk_overlap >= s.chunk_words:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_WORDS")
    if s.retrieval_mode not in {"dense", "bm25", "hybrid"}:
        raise ConfigError("RETRIEVAL_MODE must be dense, bm25 or hybrid")
    if not 1 <= s.agent_max_steps <= 20:
        raise ConfigError("AGENT_MAX_STEPS must be between 1 and 20")
    return s
