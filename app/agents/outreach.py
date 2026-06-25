"""Outreach agent — drafts a personalised invitation for a shortlisted expert."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.agents import prompts
from app.agents.base import run_stage
from app.agents.llm_client import LLMClient
from app.models.schemas import ExpertProfile, Match, OutreachDraft, ProjectBrief


def draft_outreach(
    profile: ExpertProfile,
    brief: ProjectBrief,
    match: Match,
    *,
    run_id: uuid.UUID,
    session: Session | None = None,
    expert_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
) -> OutreachDraft:
    prompt = prompts.render(
        "outreach",
        "v1",
        profile=profile.model_dump_json(indent=2),
        brief=brief.model_dump_json(indent=2),
        rationale=match.rationale,
    )
    return run_stage(
        stage="outreach",
        run_id=run_id,
        prompt=prompt,
        response_model=OutreachDraft,
        mock_payload={
            "profile": profile.model_dump(),
            "brief": brief.model_dump(),
            "match": match.model_dump(mode="json"),
        },
        prompt_version="v1",
        temperature=0.4,
        session=session,
        expert_id=expert_id,
        project_id=project_id,
        llm=llm,
    )
