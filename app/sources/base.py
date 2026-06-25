"""Source adapter interface.

A *source* supplies raw expert bios to ingest. ``SyntheticSource`` is the only
implementation today; a real provider (Proxycurl, Clearbit, an internal CRM
export…) is one new subclass implementing ``fetch`` — the agent pipeline that
consumes bios never changes. The data choice is pluggable; the agents are the
product.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import BioInput


class BaseSource(ABC):
    """Adapter that yields raw bios for ingestion."""

    name: str = "base"

    @abstractmethod
    def fetch(self, limit: int | None = None) -> list[BioInput]:
        """Return up to ``limit`` raw bios (all of them if ``limit`` is None)."""
        raise NotImplementedError
