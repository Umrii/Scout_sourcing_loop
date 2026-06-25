"""Pydantic v2 contracts — the structured-output backbone of Scout.

Every LLM call returns one of these, schema-validated. The same models double
as the API request/response shapes, so the structured-output contract and the
HTTP contract never drift apart.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────── enums ───────────────────────────
class Seniority(str, Enum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    exec = "exec"

    @property
    def rank(self) -> int:
        """Ordinal used by the deterministic seniority-fit calculation."""
        return _SENIORITY_RANK[self]


_SENIORITY_RANK: dict[Seniority, int] = {
    Seniority.junior: 0,
    Seniority.mid: 1,
    Seniority.senior: 2,
    Seniority.lead: 3,
    Seniority.exec: 4,
}


class Relevance(str, Enum):
    strong = "strong"
    possible = "possible"
    weak = "weak"


class EdgeCase(str, Enum):
    clean = "clean"
    missing_field = "missing_field"
    ambiguous_title = "ambiguous_title"
    malformed = "malformed"
    multi_domain = "multi_domain"


# ─────────────────────── core domain models ───────────────────────
class ExpertProfile(BaseModel):
    """Structured profile extracted from a messy free-text bio.

    Critical rule enforced by the Extract prompt + eval: information that is
    *not present* in the bio must be `null`, never invented.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str
    current_title: str | None = None
    company: str | None = None
    domains: list[str] = Field(default_factory=list)
    seniority: Seniority | None = None
    years_experience: int | None = None
    location: str | None = None
    notable_topics: list[str] = Field(default_factory=list)
    extraction_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="The model's own confidence."
    )


class ProjectBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str
    description: str
    required_domains: list[str] = Field(default_factory=list)
    min_seniority: Seniority | None = None
    num_experts_needed: int = Field(default=3, ge=1, le=50)


# ────────────────────── agent intermediate outputs ──────────────────────
class Classification(BaseModel):
    """Output of the Classify agent (LLM semantic judgement)."""

    relevance: Relevance
    domain_match_score: float = Field(ge=0.0, le=1.0)
    seniority_fit: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Enrichment(BaseModel):
    """Output of the Enrich agent + stubbed provider lookup."""

    rationale: str
    sub_specialties: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    provider_data: dict | None = None


class OutreachDraft(BaseModel):
    """Output of the Outreach agent."""

    message: str


# ─────────────────────────── persisted rows ───────────────────────────
class Expert(ExpertProfile):
    """An ExpertProfile as stored: adds identity + provenance."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    raw_bio: str
    source: str
    created_at: datetime


class Match(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID | None = None
    project_id: UUID
    expert_id: UUID
    relevance: Relevance
    domain_match_score: float
    seniority_fit: float
    overall_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    outreach_draft: str | None = None
    created_at: datetime | None = None


class Project(ProjectBrief):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime


class AgentRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    stage: str
    expert_id: UUID | None = None
    project_id: UUID | None = None
    model: str | None = None
    prompt_version: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str
    error: str | None = None
    created_at: datetime


# ─────────────────────── API request / response shapes ───────────────────────
class BioInput(BaseModel):
    raw_bio: str
    source: str = "synthetic"


class IngestRequest(BaseModel):
    bios: list[BioInput]


class SourceRunResponse(BaseModel):
    """Returned immediately by POST /projects/{id}/source (background job)."""

    run_id: UUID
    project_id: UUID
    status: str = "started"
    detail: str = "Sourcing run started; poll /projects/{id}/matches."


class MatchWithExpert(Match):
    """A Match joined with its expert, for the ranked-shortlist response."""

    expert: Expert
