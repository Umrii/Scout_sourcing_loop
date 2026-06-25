"""Shared agent runner.

Every agent is the same shape: render a versioned prompt, make one structured
LLM call, log a row to ``agent_runs``, return a validated object. ``run_stage``
captures that pattern (including the failure path, where telemetry must still be
recorded) so the individual agents stay one-liners.
"""
from __future__ import annotations

import uuid
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.llm_client import LLMClient, LLMError, get_llm
from app.observability.logging import CallMeta, record_agent_run

M = TypeVar("M", bound=BaseModel)


def run_stage(
    *,
    stage: str,
    run_id: uuid.UUID,
    prompt: str,
    response_model: type[M],
    mock_payload: dict | None = None,
    prompt_version: str | None = None,
    temperature: float = 0.2,
    session: Session | None = None,
    expert_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
) -> M:
    """Run one agent stage, logging to ``agent_runs`` when a session is given.

    ``session=None`` runs the stage without persisting telemetry — used by the
    eval harness, which scores extraction in bulk and tracks its own metrics.
    """
    llm = llm or get_llm()
    try:
        obj, meta = llm.complete_json(
            prompt=prompt,
            response_model=response_model,
            mock_payload=mock_payload,
            prompt_version=prompt_version,
            temperature=temperature,
        )
    except LLMError as exc:
        meta = exc.meta or CallMeta(
            model=getattr(llm, "model", llm.name),
            prompt_version=prompt_version,
            status="error",
            error=str(exc),
        )
        if session is not None:
            record_agent_run(
                session,
                run_id=run_id,
                stage=stage,
                meta=meta,
                expert_id=expert_id,
                project_id=project_id,
            )
        raise

    if session is not None:
        record_agent_run(
            session,
            run_id=run_id,
            stage=stage,
            meta=meta,
            expert_id=expert_id,
            project_id=project_id,
        )
    return obj
