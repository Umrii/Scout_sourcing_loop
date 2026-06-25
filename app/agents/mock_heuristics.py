"""Deterministic stand-ins for the LLM, used in offline / `mock` mode.

These are *not* the product — they're a reproducible substitute so the pipeline,
seed script, and tests run with no API key. They deliberately model the kind of
mistakes a weak extractor makes (and how a better prompt fixes them), which is
what lets the eval harness show a genuine v1→v2 delta offline.
"""
from __future__ import annotations

import re

from app.models.schemas import Relevance, Seniority

# ───────────────────── shared domain vocabulary ─────────────────────
# canonical domain -> trigger substrings (lowercased) found in free text
DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "carbon markets": [
        "carbon market", "carbon credit", "carbon trading", "voluntary carbon",
        "compliance carbon", "offset", "emissions trading", "cap and trade",
        "cap-and-trade", " ets", "net zero", "net-zero", "decarbon",
    ],
    "climate policy": [
        "climate policy", "climate", "energy transition", "sustainability",
        "renewable", "clean energy", "esg", "environmental policy",
    ],
    "fintech": [
        "fintech", "payment", "open banking", "lending", "neobank", "wallet",
        "card issuing", "financial services", "banking",
    ],
    "healthcare": [
        "healthcare", "health tech", "clinical", "medical", "biotech",
        "pharma", "patient", "hospital", "genomics", "diagnostics",
    ],
    "machine learning": [
        "machine learning", " ml ", "deep learning", "neural", "nlp",
        "computer vision", "data science", "llm", "recommender", "a.i.",
        "artificial intelligence",
    ],
    "cybersecurity": [
        "cybersecurity", "security", "infosec", "threat", "appsec",
        "penetration", "zero trust",
    ],
    "supply chain": [
        "supply chain", "logistics", "procurement", "freight", "warehousing",
    ],
}

_SENIORITY_PATTERNS: list[tuple[Seniority, list[str]]] = [
    (Seniority.exec, ["chief", "ceo", "cto", "cfo", "coo", "vp ", "vice president",
                      "head of", "founder", "co-founder", "partner", "president"]),
    (Seniority.lead, ["lead", "principal", "staff ", "director", "manager"]),
    (Seniority.senior, ["senior", "sr.", "sr ", "expert", "veteran"]),
    (Seniority.junior, ["junior", "jr.", "associate", "intern", "graduate",
                        "entry-level", "early-career"]),
]


# ─────────────────────────── extraction ───────────────────────────
def extract(raw_bio: str, prompt_version: str | None = None) -> dict:
    """Heuristic bio → profile. v1 is deliberately weaker than v2."""
    text = raw_bio.strip()
    low = " " + text.lower() + " "
    is_v2 = (prompt_version or "v2") != "v1"

    name = _extract_name(text, honorific_aware=is_v2)
    title = _extract_title(text)
    company = _extract_company(text)
    years = _extract_years(text, words_aware=is_v2)
    location = _extract_location(text)
    domains = _match_domains(low)
    topics = _extract_topics(low)
    seniority = _detect_seniority(low)

    # The core v1 vs v2 difference: v1 guesses "mid" when seniority is unknown
    # (a hallucination the eval punishes); v2 honestly returns null.
    if seniority is None and not is_v2:
        seniority = Seniority.mid

    # v1 also leaves years null when written as words ("a decade"); handled above.
    confidence = 0.55 if not is_v2 else 0.8
    if not domains:
        confidence -= 0.1

    return {
        "name": name,
        "current_title": title,
        "company": company,
        "domains": domains,
        "seniority": seniority.value if seniority else None,
        "years_experience": years,
        "location": location,
        "notable_topics": topics,
        "extraction_confidence": round(max(0.1, min(confidence, 0.95)), 2),
    }


_NAME_DELIM = re.compile(
    r",\s+|\s+[—–-]\s+|\s+(?:is|was|who|works|serves|currently)\s+", re.I
)


def _extract_name(text: str, honorific_aware: bool) -> str:
    head = _NAME_DELIM.split(text, maxsplit=1)[0].strip().strip(",")
    if honorific_aware:
        head = re.sub(r"^(dr|mr|mrs|ms|prof|professor)\.?\s+", "", head, flags=re.I)
    words = head.split()
    if 1 <= len(words) <= 4 and head[:1].isupper():
        return head
    # fallback: leading run of capitalised tokens (stops at the first lowercase
    # word, so "Sam runs growth" -> "Sam", not "Sam Worked")
    lead: list[str] = []
    for w in text.split():
        if w[:1].isupper():
            lead.append(w.strip(".,"))
        else:
            break
    return " ".join(lead[:3]) or (words[0] if words else "Unknown")


def _extract_title(text: str) -> str | None:
    m = re.search(
        r"\b(?:is|was|as|serves as|works as)\s+(?:an?|the)\s+"
        r"([A-Za-z][\w/&\- ]{2,40}?)\s+(?:at|@|for|with|in)\b",
        text,
    )
    if m:
        return _clean(m.group(1))
    m = re.search(
        r"\b((?:Senior |Staff |Principal |Lead |Chief |Head of |VP of |Director of )"
        r"[\w/&\- ]{2,40}?)(?:\s+(?:at|@|for|with)\b|[\.,])",
        text,
    )
    if m:
        return _clean(m.group(1))
    # "Name, Title, Company" pattern
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2 and 1 <= len(parts[1].split()) <= 6:
        cand = parts[1]
        if any(k in cand.lower() for k in ("engineer", "scientist", "manager",
                                            "analyst", "lead", "director", "officer",
                                            "researcher", "consultant", "advisor")):
            return _clean(cand)
    return None


def _extract_company(text: str) -> str | None:
    m = re.search(
        r"\b(?:at|@|with|for)\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})",
        text,
    )
    if m:
        company = m.group(1).split(".")[0]  # cut at a sentence boundary
        # drop trailing pronouns/connectives the greedy match may grab
        company = re.sub(
            r"\s+(?:She|He|They|Where|And|In|Based|Since|For|The)\b.*$", "", company
        )
        return _clean(company) or None
    return None


def _extract_years(text: str, words_aware: bool) -> int | None:
    m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)", text, flags=re.I)
    if m:
        return int(m.group(1))
    if words_aware:
        words = {"a decade": 10, "over a decade": 12, "two decades": 20,
                 "fifteen": 15, "twenty": 20, "ten": 10, "five": 5, "eight": 8}
        low = text.lower()
        for phrase, val in words.items():
            if phrase in low:
                return val
    return None


def _extract_location(text: str) -> str | None:
    m = re.search(
        r"\b(?:based in|located in|lives in|based out of)\s+"
        r"([A-Za-z][\w .,'\-]+?)(?:[\.;,]|\s+(?:and|where|with)\b|$)",
        text,
        flags=re.I,
    )
    if m:
        return _clean(m.group(1))
    return None


def _match_domains(low_text: str) -> list[str]:
    found: list[str] = []
    for domain, triggers in DOMAIN_SYNONYMS.items():
        if any(t in low_text for t in triggers):
            found.append(domain)
    return found


def _extract_topics(low_text: str) -> list[str]:
    topics: list[str] = []
    for triggers in DOMAIN_SYNONYMS.values():
        for t in triggers:
            t = t.strip()
            if len(t) > 4 and t in low_text and t not in topics:
                topics.append(t)
    return topics[:5]


def _detect_seniority(low_text: str) -> Seniority | None:
    for level, keywords in _SENIORITY_PATTERNS:
        if any(k in low_text for k in keywords):
            return level
    return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,-&")


# ─────────────────────────── classification ───────────────────────────
def classify(profile: dict, brief: dict) -> dict:
    required = [d.lower() for d in brief.get("required_domains", [])]
    have = _expand_domains(profile)

    if required:
        matched = [d for d in required if _domain_overlaps(d, have)]
        domain_score = len(matched) / len(required)
    else:
        matched = list(have)
        domain_score = 0.5 if have else 0.0

    seniority_fit = _seniority_fit(
        profile.get("seniority"), brief.get("min_seniority")
    )

    if domain_score >= 0.67 and seniority_fit >= 0.5:
        relevance = Relevance.strong
    elif domain_score >= 0.34:
        relevance = Relevance.possible
    else:
        relevance = Relevance.weak

    reasoning = (
        f"Matched {len(matched)}/{len(required) or 'n'} required domains"
        + (f" ({', '.join(matched)})" if matched else "")
        + f"; seniority {profile.get('seniority') or 'unknown'} vs "
        f"min {brief.get('min_seniority') or 'none'}."
    )
    return {
        "relevance": relevance.value,
        "domain_match_score": round(domain_score, 3),
        "seniority_fit": round(seniority_fit, 3),
        "reasoning": reasoning,
    }


def _expand_domains(profile: dict) -> set[str]:
    blob = " ".join(
        [*(profile.get("domains") or []), *(profile.get("notable_topics") or [])]
    ).lower()
    have = set(profile.get("domains") or [])
    for domain, triggers in DOMAIN_SYNONYMS.items():
        if any(t in (" " + blob + " ") for t in triggers):
            have.add(domain)
    return {h.lower() for h in have}


def _domain_overlaps(required: str, have: set[str]) -> bool:
    if required in have:
        return True
    # token overlap fallback ("carbon markets" vs "carbon trading")
    req_tokens = set(required.split())
    return any(req_tokens & set(h.split()) for h in have)


def _seniority_fit(seniority: str | None, min_seniority: str | None) -> float:
    if seniority is None:
        return 0.4  # unknown — neither rewarded nor zeroed
    rank = Seniority(seniority).rank
    if not min_seniority:
        return round(0.6 + 0.1 * rank, 3)  # mild preference for seniority
    min_rank = Seniority(min_seniority).rank
    if rank >= min_rank:
        return 1.0
    return max(0.0, 1.0 - 0.3 * (min_rank - rank))


# ─────────────────────────── enrichment ───────────────────────────
def enrich(profile: dict, brief: dict, classification: dict) -> dict:
    topics = (profile.get("notable_topics") or [])[:3] or (
        profile.get("domains") or []
    )[:3]
    focus = ", ".join(brief.get("required_domains", []) or ["the brief"])
    name = profile.get("name", "This expert")
    rationale = (
        f"{name}'s background"
        + (f" in {', '.join(topics)}" if topics else "")
        + f" maps onto {focus}. "
        + (
            f"{profile['seniority'].title()}-level seniority"
            if profile.get("seniority")
            else "Seniority is unclear from the bio"
        )
        + f", and the classifier rated domain fit at "
        f"{classification.get('domain_match_score', 0):.2f}."
    )
    confidence = round(
        0.5 * classification.get("domain_match_score", 0.5)
        + 0.5 * (profile.get("extraction_confidence") or 0.6),
        3,
    )
    return {
        "rationale": rationale,
        "sub_specialties": topics,
        "confidence": confidence,
        "provider_data": None,  # attached by the agent from the provider stub
    }


# ─────────────────────────── outreach ───────────────────────────
def outreach(profile: dict, brief: dict, match: dict) -> dict:
    first = (profile.get("name") or "there").split()[0]
    title = profile.get("current_title")
    topics = profile.get("notable_topics") or profile.get("domains") or []
    topic_phrase = f" on {topics[0]}" if topics else ""
    role_phrase = f" as {title}" if title else ""
    message = (
        f"Hi {first},\n\n"
        f"I'm putting together a small group of experts for \"{brief.get('title')}\" "
        f"and your work{role_phrase}{topic_phrase} stood out. "
        f"{match.get('rationale', '').strip()} "
        f"Would you be open to a 30-minute conversation in the next couple of weeks?\n\n"
        f"Best,\nScout"
    )
    return {"message": re.sub(r"\s+\n", "\n", message).strip()}
