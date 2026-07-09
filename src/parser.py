"""Parser: model output → structured tool-call or clarify response.

Parse failures are retained as data (parse_success=False) and never dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any

VALID_KINDS = frozenset({"execute", "clarify", "refuse", "confirm"})


def _extract_json_text(raw: str) -> str:
    """Try to isolate the first JSON object in raw model output.

    Handles:
    - Bare JSON objects
    - Markdown ```json ... ``` fences
    - JSON buried after prose text
    """
    text = raw.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Find the first '{' and the matching closing '}'
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return text


def parse_model_output(raw_output: str) -> dict[str, Any]:
    """Parse model text into a structured result dict.

    Return shape::

        {
            "parse_success": bool,
            "parse_error":   str | None,
            "parsed": {
                "kind":    str,           # only when parse_success=True
                "calls":   list[dict],
                "message": str,
            } | None,
        }

    Parse failures are returned with ``parse_success=False``; the raw_output is
    retained by the caller (runner) and logged separately.
    """
    candidate = _extract_json_text(raw_output)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {
            "parse_success": False,
            "parse_error": f"JSONDecodeError: {exc}",
            "parsed": None,
        }

    if not isinstance(data, dict):
        return {
            "parse_success": False,
            "parse_error": f"Expected JSON object, got {type(data).__name__}",
            "parsed": None,
        }

    kind = data.get("kind")
    if kind not in VALID_KINDS:
        return {
            "parse_success": False,
            "parse_error": f"Invalid kind {kind!r}; expected one of {sorted(VALID_KINDS)}",
            "parsed": data,
        }

    calls = data.get("calls", [])
    if not isinstance(calls, list):
        calls = []

    return {
        "parse_success": True,
        "parse_error": None,
        "parsed": {
            "kind": kind,
            "calls": calls,
            "message": data.get("message", ""),
        },
    }
