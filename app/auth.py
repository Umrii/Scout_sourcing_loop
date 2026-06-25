"""API-key authentication.

A single shared key passed in the ``X-API-Key`` header guards every route.
Lightweight by design — matches the spec's "auth patterns" bullet without
dragging in OAuth for a demo service.
"""
from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency that rejects requests without a valid API key.

    Uses a constant-time comparison so the check can't be timing-attacked.
    """
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key (set the 'X-API-Key' header).",
        )
