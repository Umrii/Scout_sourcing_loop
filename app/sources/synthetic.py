"""SyntheticSource — reads the generated synthetic expert pool.

Synthetic data is a deliberate choice, not a shortcut: we control ground truth
(so the eval has an answer key), there are no privacy/scraping concerns on a
public demo, and we can manufacture exactly the messiness the eval needs.

The pool file is produced by ``scripts/generate_synthetic.py``. Each line carries
the messy ``raw_bio`` *and* the clean structured ``expert`` (ground truth) and an
``edge_case`` tag. ``fetch`` exposes the universal bio contract; ``fetch_records``
exposes the synthetic-only ground truth used by seeding and the eval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import BioInput, EdgeCase, ExpertProfile
from app.sources.base import BaseSource

DEFAULT_POOL_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic_pool.jsonl"


@dataclass
class SyntheticRecord:
    raw_bio: str
    expert: ExpertProfile  # clean ground-truth profile
    edge_case: EdgeCase
    source: str = "synthetic"


class SyntheticSource(BaseSource):
    name = "synthetic"

    def __init__(self, pool_path: Path | str = DEFAULT_POOL_PATH) -> None:
        self.pool_path = Path(pool_path)

    def _read(self) -> list[SyntheticRecord]:
        if not self.pool_path.exists():
            raise FileNotFoundError(
                f"Synthetic pool not found at {self.pool_path}. "
                "Run `python -m scripts.generate_synthetic` first."
            )
        records: list[SyntheticRecord] = []
        for line in self.pool_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                SyntheticRecord(
                    raw_bio=obj["raw_bio"],
                    expert=ExpertProfile.model_validate(obj["expert"]),
                    edge_case=EdgeCase(obj.get("edge_case", "clean")),
                    source=obj.get("source", "synthetic"),
                )
            )
        return records

    def fetch(self, limit: int | None = None) -> list[BioInput]:
        records = self._read()
        if limit is not None:
            records = records[:limit]
        return [BioInput(raw_bio=r.raw_bio, source=r.source) for r in records]

    def fetch_records(self, limit: int | None = None) -> list[SyntheticRecord]:
        """Synthetic-only: bios *with* their ground-truth profiles."""
        records = self._read()
        return records[:limit] if limit is not None else records
