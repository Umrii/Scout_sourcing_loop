"""Structured observability.

Every agent call emits exactly one JSON log line *and* one row in ``agent_runs``
(surfaced at ``GET /runs``). Same discipline whether the LLM call hit Gemini or
the deterministic mock — so latency/token/version data is always queryable.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db import AgentRunRow

# ──────────────────────── JSON line logger ────────────────────────
_logger = logging.getLogger("scout")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


@dataclass
class CallMeta:
    """Telemetry returned by every LLM call (real or mock)."""

    model: str
    prompt_version: str | None = None
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "ok"
    error: str | None = None
    extra: dict = field(default_factory=dict)


def log_event(event: str, **fields) -> None:
    """Emit a single structured JSON log line to stdout."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{k: _json_safe(v) for k, v in fields.items() if v is not None},
    }
    _logger.info(json.dumps(record))


def record_agent_run(
    session: Session,
    *,
    run_id: uuid.UUID,
    stage: str,
    meta: CallMeta,
    expert_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
) -> None:
    """Persist one ``agent_runs`` row and emit the matching JSON log line.

    Uses the caller's session but flushes immediately so the row is visible
    even if a later stage in the same run fails.
    """
    row = AgentRunRow(
        run_id=run_id,
        stage=stage,
        expert_id=expert_id,
        project_id=project_id,
        model=meta.model,
        prompt_version=meta.prompt_version,
        latency_ms=meta.latency_ms,
        input_tokens=meta.input_tokens,
        output_tokens=meta.output_tokens,
        status=meta.status,
        error=meta.error,
    )
    session.add(row)
    session.flush()

    log_event(
        "agent_run",
        run_id=str(run_id),
        stage=stage,
        expert_id=str(expert_id) if expert_id else None,
        project_id=str(project_id) if project_id else None,
        model=meta.model,
        prompt_version=meta.prompt_version,
        latency_ms=meta.latency_ms,
        input_tokens=meta.input_tokens,
        output_tokens=meta.output_tokens,
        status=meta.status,
        error=meta.error,
    )


def _json_safe(value):
    if isinstance(value, (uuid.UUID, datetime)):
        return str(value)
    return value
