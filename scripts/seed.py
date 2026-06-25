"""Seed the database with the synthetic expert pool.

Default mode runs the bios through the real **Extract** agent (so the seeded
profiles are exactly what the pipeline produces — works offline in mock mode).
``--direct`` loads the clean ground-truth profiles instead, for a pristine demo.

    python -m scripts.seed                 # extract bios into experts
    python -m scripts.seed --direct        # load ground-truth profiles
    python -m scripts.seed --reset         # wipe experts first
"""
from __future__ import annotations

import argparse
import uuid

from app.agents.extract import extract_profile
from app.agents.llm_client import LLMError
from app.db import ExpertRow, init_db, session_scope
from app.mappers import profile_to_expert_row
from app.sources.synthetic import DEFAULT_POOL_PATH, SyntheticSource
from scripts.generate_synthetic import generate, write_pool


def ensure_pool() -> None:
    if not DEFAULT_POOL_PATH.exists():
        path = write_pool(generate())
        print(f"Generated synthetic pool -> {path}")


def seed_if_empty() -> int:
    """Direct-load the synthetic pool only if ``experts`` is empty.

    Uses the ground-truth profiles (no LLM calls), so it's cheap enough to run on
    every startup — the app calls this when ``SEED_ON_START`` is set. Returns the
    number of experts inserted (0 if already populated).
    """
    init_db()
    ensure_pool()
    with session_scope() as session:
        if session.query(ExpertRow).count():
            return 0
        records = SyntheticSource().fetch_records()
        for r in records:
            session.add(
                profile_to_expert_row(r.expert, raw_bio=r.raw_bio, source=r.source)
            )
        return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed experts from the synthetic pool")
    parser.add_argument("--direct", action="store_true",
                        help="load ground-truth profiles instead of extracting")
    parser.add_argument("--reset", action="store_true", help="wipe experts first")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    init_db()
    ensure_pool()
    source = SyntheticSource()

    with session_scope() as session:
        if args.reset:
            deleted = session.query(ExpertRow).delete()
            print(f"Reset: removed {deleted} experts")
        elif session.query(ExpertRow).count():
            n = session.query(ExpertRow).count()
            print(f"{n} experts already present; use --reset to reload.")
            return

        if args.direct:
            records = source.fetch_records(limit=args.limit)
            for r in records:
                session.add(
                    profile_to_expert_row(r.expert, raw_bio=r.raw_bio, source=r.source)
                )
            count = len(records)
        else:
            run_id = uuid.uuid4()
            count = 0
            for bio in source.fetch(limit=args.limit):
                expert_id = uuid.uuid4()
                try:
                    profile = extract_profile(
                        bio.raw_bio, run_id=run_id, session=session, expert_id=expert_id
                    )
                except LLMError:
                    continue
                session.add(
                    profile_to_expert_row(
                        profile, raw_bio=bio.raw_bio, source=bio.source, id=expert_id
                    )
                )
                count += 1

    mode = "direct" if args.direct else "extract"
    print(f"Seeded {count} experts (mode={mode}).")


if __name__ == "__main__":
    main()
