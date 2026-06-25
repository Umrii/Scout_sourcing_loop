"""Route step — deterministic scoring + ranking (no LLM).

    overall_score = 0.6 * domain_match_score
                  + 0.3 * seniority_fit
                  + 0.1 * recency_factor

Being deterministic makes this unit-testable with no mocking, and keeps the
final ranking explainable. The LLM provides the judgement (domain/seniority
scores); the maths here is transparent and auditable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.models.schemas import Classification, Relevance


@dataclass(frozen=True)
class Weights:
    domain: float = 0.6
    seniority: float = 0.3
    recency: float = 0.1

    def __post_init__(self) -> None:
        total = self.domain + self.seniority + self.recency
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")


DEFAULT_WEIGHTS = Weights()


@dataclass
class RouteCandidate:
    expert_id: uuid.UUID
    classification: Classification
    years_experience: int | None = None


@dataclass
class ScoredCandidate:
    expert_id: uuid.UUID
    relevance: Relevance
    domain_match_score: float
    seniority_fit: float
    recency_factor: float
    overall_score: float
    rationale: str


def recency_factor(years_experience: int | None) -> float:
    """Experience-based proxy for 'recently active / established', in [0, 1].

    A real deployment would use last-activity dates from the enrichment
    provider; with synthetic data we use a bounded experience signal.
    """
    if years_experience is None:
        return 0.5
    return min(max(years_experience, 0), 15) / 15


def score_candidate(
    candidate: RouteCandidate, weights: Weights = DEFAULT_WEIGHTS
) -> ScoredCandidate:
    cls = candidate.classification
    recency = recency_factor(candidate.years_experience)
    overall = (
        weights.domain * cls.domain_match_score
        + weights.seniority * cls.seniority_fit
        + weights.recency * recency
    )
    return ScoredCandidate(
        expert_id=candidate.expert_id,
        relevance=cls.relevance,
        domain_match_score=cls.domain_match_score,
        seniority_fit=cls.seniority_fit,
        recency_factor=round(recency, 4),
        overall_score=round(overall, 4),
        rationale=cls.reasoning,
    )


def route(
    candidates: list[RouteCandidate],
    num_experts_needed: int,
    weights: Weights = DEFAULT_WEIGHTS,
) -> list[ScoredCandidate]:
    """Score all candidates, rank descending, return the top ``num_experts_needed``.

    Ties break by ``expert_id`` for a stable, reproducible ordering.
    """
    scored = [score_candidate(c, weights) for c in candidates]
    scored.sort(key=lambda s: (-s.overall_score, str(s.expert_id)))
    return scored[: max(num_experts_needed, 0)]
