"""Pydantic <-> ORM conversions.

Reading rows into Pydantic models is handled by ``Model.model_validate(row)``
(every response model sets ``from_attributes=True``). The one direction that
needs care is *writing*: enums must become their string values for the text
columns. ``profile_to_expert_row`` centralises that so ingest and the seed script
stay in sync.
"""
from __future__ import annotations

import uuid

from app.db import ExpertRow
from app.models.schemas import ExpertProfile


def profile_to_expert_row(
    profile: ExpertProfile,
    *,
    raw_bio: str,
    source: str,
    id: uuid.UUID | None = None,
) -> ExpertRow:
    return ExpertRow(
        id=id or uuid.uuid4(),
        name=profile.name,
        current_title=profile.current_title,
        company=profile.company,
        domains=list(profile.domains),
        seniority=profile.seniority.value if profile.seniority else None,
        years_experience=profile.years_experience,
        location=profile.location,
        notable_topics=list(profile.notable_topics),
        extraction_confidence=profile.extraction_confidence,
        raw_bio=raw_bio,
        source=source,
    )
