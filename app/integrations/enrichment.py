"""External enrichment-provider integration (pluggable).

The Enrich agent calls a provider for "extra data" about an expert. Today that's
a deterministic stub; swapping in a real provider (Proxycurl, Clearbit, …) is a
single new ``BaseEnrichmentProvider`` subclass — the agent logic doesn't change.
That's the same adapter discipline as ``sources.BaseSource``.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from app.config import get_settings
from app.models.schemas import ExpertProfile


class BaseEnrichmentProvider(ABC):
    """Adapter interface for an external 'extra data' lookup."""

    name: str = "base"

    @abstractmethod
    def lookup(self, profile: ExpertProfile) -> dict:
        """Return provider data for an expert (shape is provider-specific)."""
        raise NotImplementedError


class StubEnrichmentProvider(BaseEnrichmentProvider):
    """Deterministic fake lookup. Clearly labelled — invents no real PII."""

    name = "stub"

    def lookup(self, profile: ExpertProfile) -> dict:
        slug = profile.name.lower().replace(" ", "-")
        # Deterministic pseudo-metrics derived from a hash of the name, so the
        # demo is stable across runs but obviously synthetic.
        seed = int(hashlib.sha256(slug.encode()).hexdigest(), 16)
        return {
            "provider": self.name,
            "synthetic": True,
            "profile_url": f"https://example.org/experts/{slug}",
            "estimated_response_rate": round(0.2 + (seed % 60) / 100, 2),
            "recent_activity": ["conference talk", "paper", "panel"][seed % 3],
        }


_PROVIDERS: dict[str, type[BaseEnrichmentProvider]] = {
    "stub": StubEnrichmentProvider,
}


def get_enrichment_provider() -> BaseEnrichmentProvider:
    name = get_settings().enrichment_provider
    provider_cls = _PROVIDERS.get(name, StubEnrichmentProvider)
    return provider_cls()
