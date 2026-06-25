"""FastAPI application entry point.

Wires the routers, opens the DB, and exposes an unauthenticated ``/health``
liveness probe. Everything else sits behind the ``X-API-Key`` dependency.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import init_db
from app.observability.logging import log_event
from app.routers import experts, matches, projects, runs

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    if settings.seed_on_start:
        try:
            from scripts.seed import seed_if_empty  # lazy: avoids import cycle

            added = seed_if_empty()
            if added:
                log_event("seed_on_start", count=added)
        except Exception as exc:  # noqa: BLE001 — never let seeding crash startup
            log_event("seed_on_start_failed", error=str(exc))
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

# Permissive CORS: the bundled UI is same-origin, but this also lets the page be
# served from a different host during development. Mutations stay protected by
# the API key, and all data is synthetic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experts.router)
app.include_router(projects.router)
app.include_router(matches.router)
app.include_router(runs.router)

# Single-page UI, served by the same service (one deploy, no separate host).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
