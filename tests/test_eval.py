"""The eval scoring rules — the part the harness lives or dies on."""
from __future__ import annotations

from app.models.schemas import ExpertProfile
from eval.run_eval import _list_correct, _scalar_correct, evaluate, load_dataset, score_record

EXPECTED_FULL = {
    "name": "Lena Park",
    "current_title": "Senior Climate Economist",
    "company": "Meridian",
    "domains": ["carbon markets"],
    "seniority": "senior",
    "years_experience": 12,
    "location": "Geneva",
}


def test_null_equals_null_is_correct():
    profile = ExpertProfile(name="Lena Park", company=None)
    score = score_record(profile, {"name": "Lena Park", "company": None, "domains": []})
    assert score["fields"]["company"] is True


def test_hallucination_is_penalised_and_tracked():
    profile = ExpertProfile(name="Lena Park", company="InventedCorp")
    score = score_record(profile, {"name": "Lena Park", "company": None, "domains": []})
    assert score["fields"]["company"] is False
    assert "company" in score["hallucinations"]


def test_scalar_match_is_case_insensitive():
    assert _scalar_correct("Meridian", "meridian ")
    assert not _scalar_correct("Meridian", "Other")


def test_list_match_uses_overlap_threshold():
    assert _list_correct(["carbon markets", "climate policy"], ["carbon markets"])
    assert not _list_correct(["fintech"], ["healthcare"])


def test_v2_beats_v1_on_golden_set():
    """The headline regression guard: v2 must not score below v1."""
    dataset = load_dataset()
    v1 = evaluate("v1", dataset)
    v2 = evaluate("v2", dataset)
    assert v2["extraction_accuracy"] >= v1["extraction_accuracy"]
    assert v2["hallucination_rate"] <= v1["hallucination_rate"]
