"""Route is deterministic — no LLM mock needed, just maths."""
from __future__ import annotations

import uuid

import pytest

from app.agents.route import (
    DEFAULT_WEIGHTS,
    RouteCandidate,
    Weights,
    recency_factor,
    route,
    score_candidate,
)
from app.models.schemas import Classification, Relevance


def _cls(domain: float, seniority: float, rel: Relevance = Relevance.strong) -> Classification:
    return Classification(
        relevance=rel, domain_match_score=domain, seniority_fit=seniority, reasoning="t"
    )


def _candidate(domain, seniority, years=10) -> RouteCandidate:
    return RouteCandidate(uuid.uuid4(), _cls(domain, seniority), years)


def test_weighted_formula():
    s = score_candidate(RouteCandidate(uuid.uuid4(), _cls(1.0, 1.0), 15))
    # 0.6*1 + 0.3*1 + 0.1*1 (recency capped at 15 -> 1.0)
    assert s.overall_score == 1.0


def test_weighting_prioritises_domain():
    domain_strong = score_candidate(_candidate(1.0, 0.0, 0)).overall_score
    seniority_strong = score_candidate(_candidate(0.0, 1.0, 0)).overall_score
    assert domain_strong > seniority_strong  # domain weight (0.6) > seniority (0.3)


@pytest.mark.parametrize(
    "years,expected", [(None, 0.5), (0, 0.0), (15, 1.0), (30, 1.0), (7.5, 0.5)]
)
def test_recency_factor(years, expected):
    assert recency_factor(years) == expected


def test_route_ranks_descending_and_limits():
    best = _candidate(0.9, 0.9, 12)
    mid = _candidate(0.5, 0.5, 6)
    worst = _candidate(0.1, 0.1, 1)
    top = route([worst, best, mid], num_experts_needed=2)
    assert len(top) == 2
    assert top[0].expert_id == best.expert_id
    assert top[0].overall_score >= top[1].overall_score


def test_route_zero_needed_returns_empty():
    assert route([_candidate(1.0, 1.0)], 0) == []


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        Weights(domain=0.5, seniority=0.3, recency=0.1)
    assert DEFAULT_WEIGHTS.domain == 0.6
