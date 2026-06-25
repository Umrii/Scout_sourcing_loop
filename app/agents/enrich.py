"""Enrich agent — "why they fit" + inferred sub-specialties + provider lookup.

Combines an LLM rationale with a call to the external enrichment provider,
demonstrating both the GenAI layer and external-system integration in one stage.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.agents import prompts
from app.agents.base import run_stage
from app.agents.llm_client import LLMClient
from app.integrations.enrichment import (
    BaseEnrichmentProvider,
    get_enrichment_provider,
)
from app.models.schemas import (
    Classification,
    Enrichment,
    ExpertProfile,
    ProjectBrief,
)


def enrich_match(
    profile: ExpertProfile,
    brief: ProjectBrief,
    classification: Classification,
    *,
    run_id: uuid.UUID,
    session: Session | None = None,
    expert_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
    provider: BaseEnrichmentProvider | None = None,
) -> Enrichment:
    provider = provider or get_enrichment_provider()
    provider_data = provider.lookup(profile)

    prompt = prompts.render(
        "enrich",
        "v1",
        profile=profile.model_dump_json(indent=2),
        brief=brief.model_dump_json(indent=2),
        classification=classification.model_dump_json(indent=2),
    )
    enrichment = run_stage(
        stage="enrich",
        run_id=run_id,
        prompt=prompt,
        response_model=Enrichment,
        mock_payload={
            "profile": profile.model_dump(),
            "brief": brief.model_dump(),
            "classification": classification.model_dump(),
        },
        prompt_version="v1",
        temperature=0.2,
        session=session,
        expert_id=expert_id,
        project_id=project_id,
        llm=llm,
    )
    # Provider data is integration output, not an LLM guess — attach it here.
    enrichment.provider_data = provider_data
    return enrichment
