"""FastAPI application entry point.

Wires the routers, opens the DB, and exposes an unauthenticated ``/health``
liveness probe. Everything else sits behind the ``X-API-Key`` dependency.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import init_db
from app.observability.logging import log_event
from app.routers import experts, matches, projects, runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    log_event(
        "startup",
        llm_mode=settings.resolved_llm_mode,
        model=settings.gemini_model,
        database="postgres" if settings.is_postgres else "sqlite",
    )
    yield


app = FastAPI(
    title="Scout — AI Expert-Sourcing Agent",
    version="0.1.0",
    description=(
        "A miniature of ProNexus's core sourcing loop: "
        "extract → classify → enrich → route → outreach, over a queryable "
        "org memory, with a structured-output agent layer and an eval harness."
    ),
    lifespan=lifespan,
)

app.include_router(experts.router)
app.include_router(projects.router)
app.include_router(matches.router)
app.include_router(runs.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe (no auth)."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": "scout",
        "llm_mode": settings.resolved_llm_mode,
        "database": "postgres" if settings.is_postgres else "sqlite",
    }
