"""Loader for the versioned prompt files in ``prompts/``.

Prompts are plain text with ``<<TOKEN>>`` placeholders (not ``str.format``, so the
JSON braces in few-shot examples don't need escaping). Keeping prompts as files —
not string literals — is what makes prompt-version tracking in the eval real.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=64)
def load(name: str, version: str) -> str:
    """Return the raw text of ``prompts/{name}_{version}.txt``."""
    path = PROMPTS_DIR / f"{name}_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, version: str, **subs: str) -> str:
    """Load a prompt and substitute ``<<KEY>>`` tokens (KEY = upper-cased arg)."""
    text = load(name, version)
    for key, value in subs.items():
        text = text.replace(f"<<{key.upper()}>>", str(value))
    return text
