"""Extract agent — the reliability core.

Messy free-text bio -> strict, schema-validated ``ExpertProfile``. The hard rule
(missing info => null, never invented) lives in the v2 prompt and is what the
eval harness measures.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.agents import prompts
from app.agents.base import run_stage
from app.agents.llm_client import LLMClient
from app.config import get_settings
from app.models.schemas import ExpertProfile


def extract_profile(
    raw_bio: str,
    *,
    run_id: uuid.UUID,
    session: Session | None = None,
    expert_id: uuid.UUID | None = None,
    prompt_version: str | None = None,
    llm: LLMClient | None = None,
) -> ExpertProfile:
    """Extract a structured profile from one bio.

    ``prompt_version`` defaults to the configured runtime version; the eval
    passes it explicitly to compare v1 vs v2.
    """
    version = prompt_version or get_settings().extract_prompt_version
    prompt = prompts.render("extract", version, raw_bio=raw_bio)
    return run_stage(
        stage="extract",
        run_id=run_id,
        prompt=prompt,
        response_model=ExpertProfile,
        mock_payload={"raw_bio": raw_bio},
        prompt_version=version,
        temperature=0.0,
        session=session,
        expert_id=expert_id,
        llm=llm,
    )
