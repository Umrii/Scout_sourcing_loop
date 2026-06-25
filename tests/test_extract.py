"""Extract agent: the null-discipline contract and structured-output handling."""
from __future__ import annotations

import uuid

import pytest

from app.agents.extract import extract_profile
from app.agents.llm_client import LLMError
from app.models.schemas import ExpertProfile, Seniority

AMBIGUOUS = "Sam runs growth at a fintech startup. Spent a few years on payments."
CLEAN = (
    "Dr. Lena Park is a Senior Climate Economist at Meridian. 12 years on "
    "voluntary carbon markets and emissions trading. Based in Geneva."
)


def test_v2_returns_null_for_unknown_seniority():
    """The core reliability rule: missing info -> null, never invented."""
    profile = extract_profile(AMBIGUOUS, run_id=uuid.uuid4(), prompt_version="v2")
    assert profile.seniority is None
    assert profile.years_experience is None


def test_v1_hallucinates_where_v2_is_honest():
    """v1 guesses 'mid' — exactly the failure the eval punishes and v2 fixes."""
    v1 = extract_profile(AMBIGUOUS, run_id=uuid.uuid4(), prompt_version="v1")
    assert v1.seniority == Seniority.mid


def test_clean_extraction():
    p = extract_profile(CLEAN, run_id=uuid.uuid4(), prompt_version="v2")
    assert p.name == "Lena Park"
    assert p.seniority == Seniority.senior
    assert p.years_experience == 12
    assert p.location == "Geneva"
    assert "carbon markets" in p.domains


def test_injected_llm_is_used(fake_llm):
    canned = ExpertProfile(name="Canned Person", seniority=Seniority.lead)
    fake_llm({ExpertProfile: canned})
    p = extract_profile("anything", run_id=uuid.uuid4())
    assert p.name == "Canned Person"
    assert p.seniority == Seniority.lead


def test_extraction_failure_propagates_and_logs(fake_llm):
    """A failed call raises LLMError and writes an error row to agent_runs."""
    from app.db import AgentRunRow, session_scope

    fake_llm({}, fail=True)
    run_id = uuid.uuid4()
    with session_scope() as session:
        with pytest.raises(LLMError):
            extract_profile("x", run_id=run_id, session=session, expert_id=uuid.uuid4())

    with session_scope() as session:
        row = session.query(AgentRunRow).filter(AgentRunRow.run_id == run_id).one()
        assert row.status == "error"
        assert row.stage == "extract"
