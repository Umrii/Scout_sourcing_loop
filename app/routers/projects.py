"""Project routes: create a brief, kick off the sourcing loop, read matches."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.db import ExpertRow, MatchRow, ProjectRow, get_db
from app.models.schemas import (
    Expert,
    Match,
    MatchWithExpert,
    Project,
    ProjectBrief,
    SourceRunResponse,
)
from app.pipeline import run_sourcing
from app.observability.logging import log_event

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=Project, status_code=201)
def create_project(brief: ProjectBrief, session: Session = Depends(get_db)) -> Project:
    row = ProjectRow(
        title=brief.title,
        description=brief.description,
        required_domains=list(brief.required_domains),
        min_seniority=brief.min_seniority.value if brief.min_seniority else None,
        num_experts_needed=brief.num_experts_needed,
    )
    session.add(row)
    session.commit()
    return Project.model_validate(row)


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: uuid.UUID, session: Session = Depends(get_db)) -> Project:
    row = session.get(ProjectRow, project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project.model_validate(row)


@router.post("/{project_id}/source", response_model=SourceRunResponse, status_code=202)
def source_project(
    project_id: uuid.UUID,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
) -> SourceRunResponse:
    """Kick off the sourcing loop as a background job; return a ``run_id`` now."""
    if session.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = uuid.uuid4()
    background.add_task(run_sourcing, project_id, run_id)
    log_event("source_enqueued", run_id=str(run_id), project_id=str(project_id))
    return SourceRunResponse(run_id=run_id, project_id=project_id)


@router.get("/{project_id}/matches", response_model=list[MatchWithExpert])
def get_matches(
    project_id: uuid.UUID, session: Session = Depends(get_db)
) -> list[MatchWithExpert]:
    """Ranked shortlist for a project (each match joined with its expert)."""
    if session.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = (
        select(MatchRow)
        .where(MatchRow.project_id == project_id)
        .order_by(MatchRow.overall_score.desc())
    )
    matches = session.scalars(stmt).all()

    out: list[MatchWithExpert] = []
    for m in matches:
        expert_row = session.get(ExpertRow, m.expert_id)
        if expert_row is None:
            continue
        base = Match.model_validate(m)
        out.append(
            MatchWithExpert(**base.model_dump(), expert=Expert.model_validate(expert_row))
        )
    return out
