"""LLM judge — clarify quality only (explicitly scoped; no other use)."""

from __future__ import annotations

from typing import Any


def judge_clarify_quality(
    *,
    endpoint: str,
    model: str,
    utterance: str,
    clarify_response: str,
    seed: int,
) -> dict[str, Any]:
    """Score clarify response quality via local Ollama judge."""
    ...
