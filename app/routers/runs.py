"""Observability route: recent agent runs from ``agent_runs``."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.db import AgentRunRow, get_db
from app.models.schemas import AgentRun

router = APIRouter(prefix="/runs", tags=["observability"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=list[AgentRun])
def list_runs(
    stage: str | None = Query(default=None, description="extract|classify|enrich|outreach"),
    status: str | None = Query(default=None, description="ok|error"),
    run_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_db),
) -> list[AgentRun]:
    stmt = select(AgentRunRow).order_by(AgentRunRow.created_at.desc())
    if stage is not None:
        stmt = stmt.where(AgentRunRow.stage == stage)
    if status is not None:
        stmt = stmt.where(AgentRunRow.status == status)
    if run_id is not None:
        stmt = stmt.where(AgentRunRow.run_id == run_id)
    rows = session.scalars(stmt.limit(limit)).all()
    return [AgentRun.model_validate(r) for r in rows]
