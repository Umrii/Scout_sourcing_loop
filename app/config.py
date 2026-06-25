"""Application settings, loaded from environment / `.env`.

Every setting has a local-friendly default so the service runs with zero
external dependencies (SQLite + deterministic mock LLM). Production overrides
them with Supabase + Gemini via Render env vars.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Auth ---
    api_key: str = "dev-secret-key"

    # --- Database ---
    # SQLite by default; `postgresql+psycopg://...` for Supabase/Postgres.
    database_url: str = "sqlite:///./scout.db"

    # --- LLM ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Explicit override; empty means "auto" (gemini if key present, else mock).
    llm_mode: Literal["", "gemini", "mock"] = ""

    # --- Agents ---
    extract_prompt_version: str = "v2"

    # --- Integrations ---
    enrichment_provider: str = "stub"

    @property
    def resolved_llm_mode(self) -> Literal["gemini", "mock"]:
        """Effective LLM mode after applying the auto-detection rule."""
        if self.llm_mode in ("gemini", "mock"):
            return self.llm_mode
        return "gemini" if self.gemini_api_key else "mock"

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()
