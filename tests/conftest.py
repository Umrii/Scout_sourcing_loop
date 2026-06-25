"""Test fixtures.

Env is set *before* any app import so the engine binds to an isolated SQLite
file and the LLM runs in deterministic mock mode. A ``FakeLLM`` is provided for
tests that need to pin exact structured responses or simulate failures.
"""
from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_scout.db"
os.environ["API_KEY"] = "test-key"
os.environ["LLM_MODE"] = "mock"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.agents.llm_client import LLMClient, LLMError, set_llm_override  # noqa: E402
from app.observability.logging import CallMeta  # noqa: E402

API_KEY = "test-key"


class FakeLLM(LLMClient):
    """Deterministic client that returns canned objects keyed by response model."""

    name = "fake"

    def __init__(self, responses: dict, *, fail: bool = False) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[str] = []

    def complete_json(self, *, prompt, response_model, mock_payload=None,
                      system=None, prompt_version=None, temperature=0.2):
        self.calls.append(response_model.__name__)
        meta = CallMeta(model="fake", prompt_version=prompt_version,
                        input_tokens=10, output_tokens=5)
        if self.fail:
            meta.status = "error"
            meta.error = "simulated failure"
            raise LLMError(meta.error, meta)
        value = self.responses[response_model]
        obj = value(mock_payload) if callable(value) else value
        if isinstance(obj, dict):
            obj = response_model.model_validate(obj)
        return obj, meta


@pytest.fixture(autouse=True, scope="session")
def _schema():
    """Create a clean schema once for the whole test session."""
    from app.db import Base, engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def fake_llm():
    """Install a FakeLLM; factory returns the instance and auto-clears after."""
    installed: list[FakeLLM] = []

    def _install(responses: dict, *, fail: bool = False) -> FakeLLM:
        llm = FakeLLM(responses, fail=fail)
        set_llm_override(llm)
        installed.append(llm)
        return llm

    yield _install
    set_llm_override(None)


@pytest.fixture
def client():
    """A TestClient over a freshly-reset database, pre-authenticated."""
    from app.db import Base, engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from app.main import app

    with TestClient(app) as c:
        c.headers.update({"X-API-Key": API_KEY})
        yield c
