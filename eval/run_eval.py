"""Eval harness — the centrepiece.

Scores an Extract prompt version against the golden set, field by field:

* **Field-level accuracy** — correct fields / total fields. ``null == null`` is
  correct (honest "not present" is rewarded).
* **Hallucination rate** — how often the model invents a value where the truth is
  null. Tracked separately because a system that fills every field looks
  "complete" while quietly making things up.
* **Per-field accuracy** — which fields the model is worst at.
* **Edge-case slices** — accuracy per ``edge_case`` tag.

Every run is written to ``eval_results`` and dumped to ``eval/results/``. Running
two versions prints the v1 -> v2 diff; a drop is flagged as a regression.

    python -m eval.run_eval --all              # v1 and v2, with the delta
    python -m eval.run_eval --version v2       # one version
    GEMINI_API_KEY=... python -m eval.run_eval --all   # against real Gemini
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.agents.extract import extract_profile
from app.agents.llm_client import LLMError, get_llm
from app.db import EvalResultRow, init_db, session_scope
from app.models.schemas import ExpertProfile

DATASET = Path(__file__).parent / "dataset" / "bios.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

SCALAR_FIELDS = ["name", "current_title", "company", "seniority", "years_experience", "location"]
LIST_FIELDS = ["domains"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS
LIST_MATCH_THRESHOLD = 0.5  # Jaccard overlap that counts a list field correct


# ──────────────────────────── scoring ────────────────────────────
def _norm(value):
    if value is None:
        return None
    if hasattr(value, "value"):  # enum
        value = value.value
    if isinstance(value, str):
        return " ".join(value.lower().strip().strip(".,").split())
    return value


def _scalar_correct(extracted, expected) -> bool:
    return _norm(extracted) == _norm(expected)


def _list_correct(extracted: list, expected: list) -> bool:
    a = {s.lower().strip() for s in (extracted or [])}
    b = {s.lower().strip() for s in (expected or [])}
    if not a and not b:
        return True
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= LIST_MATCH_THRESHOLD


def score_record(profile: ExpertProfile, expected: dict) -> dict:
    """Per-field correctness + hallucination flags for one record."""
    fields: dict[str, bool] = {}
    hallucinations: list[str] = []
    null_expected = 0

    for f in ALL_FIELDS:
        exp = expected.get(f)
        ext = getattr(profile, f)
        is_list = f in LIST_FIELDS
        exp_is_empty = exp is None or (is_list and not exp)
        ext_is_empty = ext is None or (is_list and not ext)

        if is_list:
            correct = _list_correct(ext, exp)
        else:
            correct = _scalar_correct(ext, exp)
        fields[f] = correct

        if exp_is_empty:
            null_expected += 1
            if not ext_is_empty:
                hallucinations.append(f)

    return {
        "fields": fields,
        "correct": sum(fields.values()),
        "total": len(fields),
        "hallucinations": hallucinations,
        "null_expected": null_expected,
    }


# ──────────────────────────── evaluation ────────────────────────────
def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(version: str, dataset: list[dict], llm=None) -> dict:
    llm = llm or get_llm()
    per_field_correct = {f: 0 for f in ALL_FIELDS}
    per_field_total = {f: 0 for f in ALL_FIELDS}
    edge_correct: dict[str, int] = {}
    edge_total: dict[str, int] = {}

    correct = total = halluc = null_expected = errors = 0

    for record in dataset:
        edge = record.get("edge_case", "clean")
        try:
            profile = extract_profile(
                record["raw_bio"], run_id=uuid.uuid4(), prompt_version=version, llm=llm
            )
        except LLMError:
            errors += 1
            continue
        s = score_record(profile, record["expected"])
        correct += s["correct"]
        total += s["total"]
        halluc += len(s["hallucinations"])
        null_expected += s["null_expected"]
        for f, ok in s["fields"].items():
            per_field_total[f] += 1
            per_field_correct[f] += int(ok)
        edge_total[edge] = edge_total.get(edge, 0) + s["total"]
        edge_correct[edge] = edge_correct.get(edge, 0) + s["correct"]

    field_acc = {
        f: round(per_field_correct[f] / per_field_total[f], 4)
        for f in ALL_FIELDS
        if per_field_total[f]
    }
    edge_acc = {
        e: round(edge_correct[e] / edge_total[e], 4) for e in edge_total if edge_total[e]
    }
    return {
        "version": version,
        "model": getattr(llm, "model", llm.name),
        "dataset_size": len(dataset),
        "extraction_accuracy": round(correct / total, 4) if total else 0.0,
        "hallucination_rate": round(halluc / null_expected, 4) if null_expected else 0.0,
        "field_accuracies": field_acc,
        "edge_case_accuracies": edge_acc,
        "errors": errors,
    }


def persist(report: dict, eval_run_id: uuid.UUID) -> None:
    with session_scope() as session:
        session.add(
            EvalResultRow(
                eval_run_id=eval_run_id,
                prompt_version=report["version"],
                model=report["model"],
                dataset_size=report["dataset_size"],
                extraction_accuracy=report["extraction_accuracy"],
                hallucination_rate=report["hallucination_rate"],
                field_accuracies=report["field_accuracies"],
                edge_case_accuracies=report["edge_case_accuracies"],
            )
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{report['version']}_{eval_run_id.hex[:8]}.json"
    out.write_text(json.dumps({**report, "eval_run_id": str(eval_run_id)}, indent=2))


# ──────────────────────────── reporting ────────────────────────────
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def print_report(report: dict) -> None:
    print(f"\n  Prompt version : {report['version']}")
    print(f"  Model          : {report['model']}")
    print(f"  Dataset size   : {report['dataset_size']}")
    print(f"  EXTRACTION ACC : {_pct(report['extraction_accuracy'])}")
    print(f"  Hallucination  : {_pct(report['hallucination_rate'])}  "
          "(value invented where truth is null)")
    print("  Per-field accuracy:")
    for f, a in report["field_accuracies"].items():
        print(f"      {f:<20} {_pct(a)}")
    print("  Per-edge-case accuracy:")
    for e, a in report["edge_case_accuracies"].items():
        print(f"      {e:<20} {_pct(a)}")
    if report["errors"]:
        print(f"  Errors         : {report['errors']}")


def print_diff(before: dict, after: dict) -> None:
    delta = after["extraction_accuracy"] - before["extraction_accuracy"]
    trend = "improvement" if delta > 0 else ("REGRESSION" if delta < 0 else "no change")
    print("\n" + "=" * 60)
    print(f"  {before['version']} -> {after['version']}:  "
          f"{_pct(before['extraction_accuracy'])} -> {_pct(after['extraction_accuracy'])}  "
          f"({delta * 100:+.1f} pts, {trend})")
    h_delta = after["hallucination_rate"] - before["hallucination_rate"]
    print(f"  hallucination: {_pct(before['hallucination_rate'])} -> "
          f"{_pct(after['hallucination_rate'])}  ({h_delta * 100:+.1f} pts)")
    if delta < 0:
        print("  [!] REGRESSION - this prompt change lowers accuracy; do not ship.")
    print("=" * 60)


def write_consolidated(path: str, reports: list[dict], backend: str, dataset_size: int) -> None:
    """Write one committed-baseline JSON: both versions + the headline delta."""
    by_version = {r["version"]: r for r in reports}
    payload = {
        "backend": backend,
        "model": reports[0]["model"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": dataset_size,
        "versions": by_version,
    }
    if "v1" in by_version and "v2" in by_version:
        v1, v2 = by_version["v1"], by_version["v2"]
        payload["headline"] = {
            "extraction_accuracy": {"v1": v1["extraction_accuracy"], "v2": v2["extraction_accuracy"]},
            "hallucination_rate": {"v1": v1["hallucination_rate"], "v2": v2["hallucination_rate"]},
            "delta_pts": round((v2["extraction_accuracy"] - v1["extraction_accuracy"]) * 100, 1),
        }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote consolidated baseline -> {out}")


# ──────────────────────────── CLI ────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Scout extraction eval harness")
    parser.add_argument("--version", default="v2", help="prompt version to score")
    parser.add_argument("--all", action="store_true",
                        help="score v1 and v2 and print the delta (the headline)")
    parser.add_argument("--no-write", action="store_true", help="skip DB/file writes")
    parser.add_argument("--out", help="write a consolidated baseline JSON to this path")
    args = parser.parse_args()

    init_db()
    dataset = load_dataset()
    llm = get_llm()
    print(f"Scout eval - backend: {llm.name}, {len(dataset)} golden records")

    versions = ["v1", "v2"] if args.all else [args.version]
    reports = []
    for v in versions:
        report = evaluate(v, dataset, llm=llm)
        print_report(report)
        if not args.no_write:
            persist(report, uuid.uuid4())
        reports.append(report)

    if len(reports) == 2:
        print_diff(reports[0], reports[1])

    if args.out:
        write_consolidated(args.out, reports, llm.name, len(dataset))


if __name__ == "__main__":
    main()
