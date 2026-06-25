"""Match routes: (re)generate the outreach draft for a single match."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agents.llm_client import LLMError
from app.agents.outreach import draft_outreach
from app.auth import require_api_key
from app.db import ExpertRow, MatchRow, ProjectRow, get_db
from app.models.schemas import ExpertProfile, Match, ProjectBrief

router = APIRouter(prefix="/matches", tags=["matches"], dependencies=[Depends(require_api_key)])


@router.post("/{match_id}/outreach", response_model=Match)
def regenerate_outreach(
    match_id: uuid.UUID, session: Session = Depends(get_db)
) -> Match:
    match_row = session.get(MatchRow, match_id)
    if match_row is None:
        raise HTTPException(status_code=404, detail="Match not found")
    expert_row = session.get(ExpertRow, match_row.expert_id)
    project_row = session.get(ProjectRow, match_row.project_id)
    if expert_row is None or project_row is None:
        raise HTTPException(status_code=404, detail="Match expert/project missing")

    profile = ExpertProfile.model_validate(expert_row)
    brief = ProjectBrief.model_validate(project_row)
    run_id = uuid.uuid4()
    try:
        draft = draft_outreach(
            profile,
            brief,
            Match.model_validate(match_row),
            run_id=run_id,
            session=session,
            expert_id=match_row.expert_id,
            project_id=match_row.project_id,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Outreach generation failed: {exc}")

    match_row.outreach_draft = draft.message
    session.commit()
    return Match.model_validate(match_row)
