"""Thin LLM wrapper with a single structured-output entry point.

`complete_json(...)` always returns a *validated Pydantic instance* plus a
`CallMeta` telemetry record. Two implementations sit behind one interface:

* `GeminiClient` — real Google Gemini structured-output calls.
* `MockLLMClient` — deterministic, offline; lets the whole pipeline, the seed
  script, and the test suite run with no API key and no network.

The mode is auto-selected (`gemini` when `GEMINI_API_KEY` is set, else `mock`)
and can be forced via `LLM_MODE`. Tests inject their own client with
`set_llm_override`.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from app.agents import mock_heuristics as mock
from app.config import get_settings
from app.models.schemas import (
    Classification,
    Enrichment,
    ExpertProfile,
    OutreachDraft,
)
from app.observability.logging import CallMeta

M = TypeVar("M", bound=BaseModel)


class LLMClient(ABC):
    """Common interface for every LLM backend."""

    name: str = "base"

    @abstractmethod
    def complete_json(
        self,
        *,
        prompt: str,
        response_model: type[M],
        mock_payload: dict | None = None,
        system: str | None = None,
        prompt_version: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[M, CallMeta]:
        """Run one structured-output call.

        `mock_payload` carries the structured inputs the offline mock needs to
        compute a deterministic answer; the real Gemini backend ignores it.
        """
        raise NotImplementedError


# ──────────────────────────── Gemini ────────────────────────────
class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        # Imported lazily so the package isn't required for offline/mock runs.
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def complete_json(
        self,
        *,
        prompt: str,
        response_model: type[M],
        mock_payload: dict | None = None,
        system: str | None = None,
        prompt_version: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[M, CallMeta]:
        from google.genai import types

        started = time.perf_counter()
        meta = CallMeta(model=self.model, prompt_version=prompt_version)
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_model,
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            obj = resp.parsed
            if obj is None:  # fall back to manual parse if SDK didn't hydrate
                obj = response_model.model_validate(json.loads(resp.text))
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                meta.input_tokens = getattr(usage, "prompt_token_count", None)
                meta.output_tokens = getattr(usage, "candidates_token_count", None)
            meta.latency_ms = _elapsed_ms(started)
            return obj, meta
        except Exception as exc:  # noqa: BLE001 — telemetry must capture all failures
            meta.status = "error"
            meta.error = f"{type(exc).__name__}: {exc}"
            meta.latency_ms = _elapsed_ms(started)
            raise LLMError(meta.error, meta) from exc


# ──────────────────────────── Mock ────────────────────────────
class MockLLMClient(LLMClient):
    """Deterministic offline backend. No network, fully reproducible."""

    name = "mock"

    # response_model -> heuristic that turns a structured payload into a dict
    _DISPATCH = {
        ExpertProfile: mock.extract,
        Classification: mock.classify,
        Enrichment: mock.enrich,
        OutreachDraft: mock.outreach,
    }

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def complete_json(
        self,
        *,
        prompt: str,
        response_model: type[M],
        mock_payload: dict | None = None,
        system: str | None = None,
        prompt_version: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[M, CallMeta]:
        started = time.perf_counter()
        meta = CallMeta(model=self.model, prompt_version=prompt_version)
        handler = self._DISPATCH.get(response_model)
        if handler is None:
            raise LLMError(f"No mock handler for {response_model.__name__}")
        payload = dict(mock_payload or {})
        # Extract's heuristic also varies by prompt version (v1 vs v2) so the
        # eval can record a genuine version-to-version delta offline.
        if response_model is ExpertProfile:
            payload["prompt_version"] = prompt_version
        raw = handler(**payload)
        obj = response_model.model_validate(raw)
        meta.input_tokens = _estimate_tokens(prompt)
        meta.output_tokens = _estimate_tokens(obj.model_dump_json())
        meta.latency_ms = _elapsed_ms(started)
        return obj, meta


class LLMError(RuntimeError):
    """Raised when a structured-output call fails (after telemetry is recorded).

    Carries the partial ``CallMeta`` so the caller can still log the failed run.
    """

    def __init__(self, message: str, meta: CallMeta | None = None) -> None:
        super().__init__(message)
        self.meta = meta


# ──────────────────────── factory / override ────────────────────────
_default: LLMClient | None = None
_override: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the active LLM client (test override wins, else cached default)."""
    if _override is not None:
        return _override
    global _default
    if _default is None:
        settings = get_settings()
        if settings.resolved_llm_mode == "gemini":
            _default = GeminiClient(settings.gemini_api_key, settings.gemini_model)
        else:
            _default = MockLLMClient()
    return _default


def set_llm_override(client: LLMClient | None) -> None:
    """Inject a client (tests) or clear it with ``None``."""
    global _override
    _override = client


def reset_llm_cache() -> None:
    """Drop the cached default so the next ``get_llm`` re-reads settings."""
    global _default
    _default = None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
