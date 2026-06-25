"""The sourcing loop, run as a background job.

For a given project: classify every expert in org memory, deterministically rank
them, then enrich + draft outreach for the shortlist and persist ``matches``.
Per-expert LLM failures are logged to ``agent_runs`` and skipped so one bad bio
never sinks the whole run.

    classify (all) -> route/rank (top N) -> enrich + outreach (shortlist)
"""
from __future__ import annotations

import uuid

from app.agents.classify import classify_expert
from app.agents.enrich import enrich_match
from app.agents.llm_client import LLMError
from app.agents.outreach import draft_outreach
from app.agents.route import RouteCandidate, route
from app.db import ExpertRow, MatchRow, ProjectRow, session_scope
from app.models.schemas import ExpertProfile, Match, ProjectBrief
from app.observability.logging import log_event


def run_sourcing(project_id: uuid.UUID, run_id: uuid.UUID) -> None:
    """Execute the full sourcing loop for one project (background entry point)."""
    log_event("sourcing_started", run_id=str(run_id), project_id=str(project_id))
    with session_scope() as session:
        project_row = session.get(ProjectRow, project_id)
        if project_row is None:
            log_event("sourcing_error", run_id=str(run_id),
                      project_id=str(project_id), error="project not found")
            return
        brief = ProjectBrief.model_validate(project_row)
        experts = session.query(ExpertRow).all()

        # Re-running a source is idempotent: clear the previous shortlist first.
        session.query(MatchRow).filter(MatchRow.project_id == project_id).delete()

        # ── stage 1: classify every expert ──
        candidates: list[RouteCandidate] = []
        profiles: dict[uuid.UUID, ExpertProfile] = {}
        classifications = {}
        for row in experts:
            profile = ExpertProfile.model_validate(row)
            try:
                cls = classify_expert(
                    profile, brief, run_id=run_id, session=session,
                    expert_id=row.id, project_id=project_id,
                )
            except LLMError:
                continue  # telemetry already recorded by the agent
            profiles[row.id] = profile
            classifications[row.id] = cls
            candidates.append(
                RouteCandidate(
                    expert_id=row.id,
                    classification=cls,
                    years_experience=row.years_experience,
                )
            )

        # ── stage 2: deterministic rank, take top N ──
        shortlist = route(candidates, brief.num_experts_needed)

        # ── stage 3: enrich + outreach for the shortlist, persist matches ──
        for scored in shortlist:
            profile = profiles[scored.expert_id]
            cls = classifications[scored.expert_id]
            rationale = cls.reasoning

            try:
                enrichment = enrich_match(
                    profile, brief, cls, run_id=run_id, session=session,
                    expert_id=scored.expert_id, project_id=project_id,
                )
                rationale = enrichment.rationale
            except LLMError:
                pass  # fall back to the classifier's reasoning

            match = Match(
                project_id=project_id,
                expert_id=scored.expert_id,
                relevance=scored.relevance,
                domain_match_score=scored.domain_match_score,
                seniority_fit=scored.seniority_fit,
                overall_score=scored.overall_score,
                rationale=rationale,
            )

            outreach_text = None
            try:
                draft = draft_outreach(
                    profile, brief, match, run_id=run_id, session=session,
                    expert_id=scored.expert_id, project_id=project_id,
                )
                outreach_text = draft.message
            except LLMError:
                pass

            session.add(
                MatchRow(
                    project_id=project_id,
                    expert_id=scored.expert_id,
                    relevance=scored.relevance.value,
                    domain_match_score=scored.domain_match_score,
                    seniority_fit=scored.seniority_fit,
                    overall_score=scored.overall_score,
                    rationale=rationale,
                    outreach_draft=outreach_text,
                )
            )

        log_event(
            "sourcing_complete",
            run_id=str(run_id),
            project_id=str(project_id),
            classified=len(candidates),
            shortlisted=len(shortlist),
        )
