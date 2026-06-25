"""Generate the synthetic expert pool (deterministic).

Two conceptual passes (per the spec): build clean structured experts, then
rewrite each into a messy free-text bio of varying quality, tagged with an
edge-case category. We do it deterministically (seeded) rather than via an LLM so
the pool — and therefore the demo — is byte-for-byte reproducible and the ground
truth is exact.

Key honesty rule (the labelling caveat): when an edge case drops a field from the
bio, the ground-truth ``expert`` for that field becomes ``null`` — the correct
answer is "not present", not the original value.

    python -m scripts.generate_synthetic            # ~45 records -> data/
    python -m scripts.generate_synthetic --n 60
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import EdgeCase, ExpertProfile, Seniority

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_pool.jsonl"

FIRST_NAMES = [
    "Lena", "Aisha", "David", "Yusuf", "Maria", "Chen", "Olusegun", "Sofia",
    "Raj", "Nadia", "Ahmed", "Elena", "Tomas", "Grace", "Liam", "Amara",
    "Hiroshi", "Fatima", "Marco", "Ingrid", "Kofi", "Sanjay", "Wei", "Noor",
]
LAST_NAMES = [
    "Park", "Khan", "Okafor", "Demir", "Gonzalez", "Wei", "Bello", "Marchetti",
    "Patel", "Hassan", "Said", "Petrova", "Alvarez", "Kim", "Nwosu", "Tanaka",
    "Rossi", "Larsen", "Mensah", "Gupta", "Haddad", "Silva", "Novak", "Yilmaz",
]
LOCATIONS = [
    "Geneva", "London", "Toronto", "Madrid", "Singapore", "Amsterdam", "Milan",
    "Bangalore", "Cairo", "Dubai", "Berlin", "Nairobi", "Tokyo", "Lagos", "Oslo",
]

DOMAINS: dict[str, dict] = {
    "carbon markets": {
        "role": "Carbon Markets Analyst",
        "dept": "Carbon Strategy",
        "companies": ["Meridian", "OffsetWorks", "CarbonChain", "NetZero Partners"],
        "topics": ["voluntary carbon markets", "emissions trading",
                   "carbon credit verification", "net-zero strategy"],
        "labels": ["carbon markets", "climate policy"],
    },
    "fintech": {
        "role": "Payments Engineer",
        "dept": "Payments",
        "companies": ["NovaBank", "PayLoop", "FinCore", "LedgerWorks"],
        "topics": ["open banking", "card issuing", "payments rails", "lending"],
        "labels": ["fintech"],
    },
    "healthcare": {
        "role": "Healthcare Data Scientist",
        "dept": "Clinical Data",
        "companies": ["VitaCare", "MedAI", "ClinIQ", "HealthBridge"],
        "topics": ["clinical diagnostics", "patient data systems",
                   "genomics", "clinical research"],
        "labels": ["healthcare"],
    },
    "machine learning": {
        "role": "Machine Learning Engineer",
        "dept": "ML Platform",
        "companies": ["Frontier AI", "DeepStack", "Inferra", "NeuralForge"],
        "topics": ["recommender systems", "NLP", "computer vision", "ML platform"],
        "labels": ["machine learning"],
    },
    "cybersecurity": {
        "role": "Security Architect",
        "dept": "Security",
        "companies": ["Sentinel Systems", "ZeroTrust Labs", "Aegis", "Bastion"],
        "topics": ["appsec", "threat modelling", "penetration testing", "zero trust"],
        "labels": ["cybersecurity"],
    },
    "supply chain": {
        "role": "Supply Chain Manager",
        "dept": "Operations",
        "companies": ["Logistica", "FreightFlow", "ChainWorks", "PortLink"],
        "topics": ["procurement", "freight", "logistics", "warehousing"],
        "labels": ["supply chain"],
    },
}

# seniority -> (title prefix builder, is exec)
SENIORITY_TITLE = {
    Seniority.junior: lambda role, dept: f"Junior {role}",
    Seniority.senior: lambda role, dept: f"Senior {role}",
    Seniority.lead: lambda role, dept: f"Lead {role}",
    Seniority.exec: lambda role, dept: f"Head of {dept}",
}
EDGE_CYCLE = [
    EdgeCase.clean, EdgeCase.clean, EdgeCase.multi_domain, EdgeCase.missing_field,
    EdgeCase.ambiguous_title, EdgeCase.malformed,
]


@dataclass
class Record:
    expert: ExpertProfile
    raw_bio: str
    edge_case: EdgeCase
    source: str = "synthetic"


def _topics_phrase(topics: list[str]) -> str:
    if len(topics) == 1:
        return topics[0]
    return ", ".join(topics[:-1]) + f" and {topics[-1]}"


def build_record(i: int, rng: random.Random) -> Record:
    domain_key = list(DOMAINS)[i % len(DOMAINS)]
    d = DOMAINS[domain_key]
    seniority = rng.choice([Seniority.junior, Seniority.senior, Seniority.lead, Seniority.exec])
    edge = EDGE_CYCLE[i % len(EDGE_CYCLE)]

    # Unique (first, last) pairing: 5 is coprime to 24, and the i//len term
    # shifts the surname on each wrap, so no two records share a name.
    last_idx = (i * 5 + i // len(FIRST_NAMES)) % len(LAST_NAMES)
    name = f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {LAST_NAMES[last_idx]}"
    company = rng.choice(d["companies"])
    location = rng.choice(LOCATIONS)
    years = rng.randint(3, 24)
    topics = rng.sample(d["topics"], k=2)
    title = SENIORITY_TITLE[seniority](d["role"], d["dept"])
    labels = list(d["labels"])

    # full ground truth; degraded per edge case below
    profile = ExpertProfile(
        name=name, current_title=title, company=company, domains=labels,
        seniority=seniority, years_experience=years, location=location,
        notable_topics=topics, extraction_confidence=None,
    )
    bio = _compose(edge, profile, d, topics, rng)
    return Record(expert=profile, raw_bio=bio, edge_case=edge)


def _compose(edge, p: ExpertProfile, d: dict, topics: list[str], rng) -> str:
    tp = _topics_phrase(topics)
    if edge is EdgeCase.clean:
        return (f"{p.name} is a {p.current_title} at {p.company}. "
                f"{p.years_experience} years working on {tp}. Based in {p.location}.")

    if edge is EdgeCase.multi_domain:
        # add a second domain's topic + label
        other_key = rng.choice([k for k in DOMAINS if k != d["labels"][0] and k not in d["labels"]])
        other = DOMAINS[other_key]
        extra_topic = rng.choice(other["topics"])
        p.domains = list(dict.fromkeys([*p.domains, other["labels"][0]]))
        p.notable_topics = [*topics, extra_topic]
        return (f"{p.name}, {p.current_title} at {p.company}, blends "
                f"{d['labels'][0]} and {other['labels'][0]} — working on {tp} and "
                f"{extra_topic}. {p.years_experience} years, based in {p.location}.")

    if edge is EdgeCase.missing_field:
        # drop company, years, location from the bio AND the ground truth
        p.company = None
        p.years_experience = None
        p.location = None
        return f"{p.name} is a {p.current_title} focused on {tp}."

    if edge is EdgeCase.ambiguous_title:
        # no explicit title -> title & seniority become null
        p.current_title = None
        p.seniority = None
        return (f"{p.name} has spent {p.years_experience} years on {tp} at "
                f"{p.company}, based in {p.location}.")

    # malformed: fragment format, no title/seniority signal
    p.current_title = None
    p.seniority = None
    frag_topics = " / ".join(topics)
    return (f"{p.name.upper()} | {frag_topics} | {p.company} | "
            f"~{p.years_experience} yrs | based in {p.location}")


def generate(n: int = 45, seed: int = 42) -> list[Record]:
    rng = random.Random(seed)
    return [build_record(i, rng) for i in range(n)]


def write_pool(records: list[Record], path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({
                "raw_bio": r.raw_bio,
                "expert": r.expert.model_dump(mode="json"),
                "edge_case": r.edge_case.value,
                "source": r.source,
            }) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic expert pool")
    parser.add_argument("--n", type=int, default=45, help="number of experts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = generate(args.n, args.seed)
    path = write_pool(records)
    by_edge: dict[str, int] = {}
    for r in records:
        by_edge[r.edge_case.value] = by_edge.get(r.edge_case.value, 0) + 1
    print(f"Wrote {len(records)} synthetic experts -> {path}")
    print("Edge-case distribution:", json.dumps(by_edge))


if __name__ == "__main__":
    main()
