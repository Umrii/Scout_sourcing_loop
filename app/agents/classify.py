"""Classify agent — semantic relevance judgement.

Given a profile and a brief, the LLM decides relevance and returns structured
scores (domain match, seniority fit) plus a one-line reasoning trace. The
numbers feed the deterministic Route step.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.agents import prompts
from app.agents.base import run_stage
from app.agents.llm_client import LLMClient
from app.models.schemas import Classification, ExpertProfile, ProjectBrief


def classify_expert(
    profile: ExpertProfile,
    brief: ProjectBrief,
    *,
    run_id: uuid.UUID,
    session: Session | None = None,
    expert_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
) -> Classification:
    prompt = prompts.render(
        "classify",
        "v1",
        profile=profile.model_dump_json(indent=2),
        brief=brief.model_dump_json(indent=2),
    )
    return run_stage(
        stage="classify",
        run_id=run_id,
        prompt=prompt,
        response_model=Classification,
        mock_payload={"profile": profile.model_dump(), "brief": brief.model_dump()},
        prompt_version="v1",
        temperature=0.1,
        session=session,
        expert_id=expert_id,
        project_id=project_id,
        llm=llm,
    )
