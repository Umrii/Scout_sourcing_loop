"""Expert routes: ingestion (runs Extract) + org-memory search."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.extract import extract_profile
from app.agents.llm_client import LLMError
from app.auth import require_api_key
from app.db import ExpertRow, get_db
from app.mappers import profile_to_expert_row
from app.models.schemas import Expert, IngestRequest, Seniority
from app.observability.logging import log_event

router = APIRouter(prefix="/experts", tags=["experts"], dependencies=[Depends(require_api_key)])


@router.post("/ingest", response_model=list[Expert])
def ingest_experts(req: IngestRequest, session: Session = Depends(get_db)) -> list[Expert]:
    """Run the Extract agent over raw bios, store the profiles, return them.

    One ``run_id`` ties the whole batch together in ``agent_runs``. A bio whose
    extraction fails is logged and skipped — the rest still land.
    """
    run_id = uuid.uuid4()
    stored: list[ExpertRow] = []
    for bio in req.bios:
        expert_id = uuid.uuid4()
        try:
            profile = extract_profile(
                bio.raw_bio, run_id=run_id, session=session, expert_id=expert_id
            )
        except LLMError:
            continue
        row = profile_to_expert_row(
            profile, raw_bio=bio.raw_bio, source=bio.source, id=expert_id
        )
        session.add(row)
        stored.append(row)

    session.commit()
    log_event("ingest", run_id=str(run_id), requested=len(req.bios), stored=len(stored))
    return [Expert.model_validate(r) for r in stored]


@router.get("/search", response_model=list[Expert])
def search_experts(
    domain: str | None = Query(default=None, description="Match a domain or topic."),
    seniority: Seniority | None = Query(default=None),
    q: str | None = Query(default=None, description="Free-text across the profile."),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_db),
) -> list[Expert]:
    """Org memory — "have we sourced carbon experts before?".

    Seniority filters in SQL; domain/free-text filter in Python for dialect
    portability. At scale the ``domains`` GIN index (see tables.sql) backs a
    jsonb-containment query instead.
    """
    stmt = select(ExpertRow)
    if seniority is not None:
        stmt = stmt.where(ExpertRow.seniority == seniority.value)
    rows = session.scalars(stmt).all()

    results = [r for r in rows if _matches(r, domain, q)][:limit]
    return [Expert.model_validate(r) for r in results]


@router.get("/{expert_id}", response_model=Expert)
def get_expert(expert_id: uuid.UUID, session: Session = Depends(get_db)) -> Expert:
    row = session.get(ExpertRow, expert_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Expert not found")
    return Expert.model_validate(row)


def _matches(row: ExpertRow, domain: str | None, q: str | None) -> bool:
    if domain:
        needle = domain.lower()
        haystack = [d.lower() for d in (row.domains or [])] + [
            t.lower() for t in (row.notable_topics or [])
        ]
        if not any(needle in h or h in needle for h in haystack):
            return False
    if q:
        blob = " ".join(
            filter(
                None,
                [
                    row.name,
                    row.current_title,
                    row.company,
                    row.location,
                    " ".join(row.domains or []),
                    " ".join(row.notable_topics or []),
                    row.raw_bio,
                ],
            )
        ).lower()
        if q.lower() not in blob:
            return False
    return True
