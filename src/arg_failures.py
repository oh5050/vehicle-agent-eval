"""Classify wrong_args failures into diagnostic categories for reporting."""

from __future__ import annotations

from typing import Any

from src.normalizer import normalize_calls
from src.registry import Registry
from src.scorer import VariantScore

ARG_FAILURE_CATEGORIES = (
    "out_of_range",
    "invalid_enum",
    "missing_required",
    "extra_arg",
    "value_mismatch",
)


def _categorize_validation(
    result,
    tool,
) -> str:
    if result.error_code == "MISSING_REQUIRED":
        return "missing_required"
    if result.error_code == "SCHEMA_VIOLATION":
        if result.field and result.field not in tool.parameters:
            return "extra_arg"
        msg = result.error_message or ""
        if "got 'NoneType'" in msg:
            return "missing_required"
        return "value_mismatch"
    if result.error_code == "OUT_OF_RANGE":
        spec = tool.parameters.get(result.field or "")
        if spec is not None and spec.enum is not None:
            return "invalid_enum"
        return "out_of_range"
    return "value_mismatch"


def classify_arg_failure(
    expected: dict[str, Any],
    parsed: dict[str, Any] | None,
    *,
    registry: Registry,
    synonyms: dict[str, dict[str, str]] | None,
    score: VariantScore,
    equivalence: bool = True,
) -> tuple[str, str]:
    """Return (category, detail) for a wrong_args case.

    Priority: schema/validation errors on raw pred → range violations → value_mismatch.
    """
    if score.failure_class != "wrong_args" or parsed is None:
        return "value_mismatch", "not wrong_args"

    pred_calls_raw: list[dict[str, Any]] = parsed.get("calls") or []
    for call in pred_calls_raw:
        tool_name = call.get("tool", "")
        args = call.get("args") or {}
        tool = registry.get_tool(tool_name)
        if tool is None:
            continue
        result = registry.validate_call(tool_name, args)
        if not result.valid:
            category = _categorize_validation(result, tool)
            return category, result.error_message or category

    pred_norm = normalize_calls(
        pred_calls_raw,
        registry=registry,
        synonyms=synonyms,
        equivalence=equivalence,
    )
    if pred_norm.violations:
        return "out_of_range", pred_norm.violations[0]

    return "value_mismatch", "schema-valid args do not match gold"


def collect_arg_failures(
    rows: list[dict[str, Any]],
    *,
    registry: Registry,
    synonyms: dict[str, dict[str, str]] | None,
    variant: str = "normalized",
) -> list[dict[str, Any]]:
    """Extract wrong_args rows with classified failure reasons."""
    records: list[dict[str, Any]] = []
    for row in rows:
        score: VariantScore = row[variant]
        if score.failure_class != "wrong_args":
            continue
        category, detail = classify_arg_failure(
            row.get("expected") or {},
            row.get("parsed"),
            registry=registry,
            synonyms=synonyms,
            score=score,
            equivalence=(variant == "normalized"),
        )
        records.append(
            {
                "record_id": row["record_id"],
                "utterance_type": row.get("utterance_type"),
                "variant": variant,
                "category": category,
                "detail": detail,
                "gold_kind": row.get("gold_kind"),
                "pred_kind": row.get("pred_kind"),
                "gold_tools": "|".join(row.get("gold_tools") or []),
                "pred_tools": "|".join(row.get("pred_tools") or []),
            }
        )
    return records


def arg_failures_to_csv(records: list[dict[str, Any]], path) -> None:
    from pathlib import Path
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "record_id",
        "utterance_type",
        "variant",
        "category",
        "detail",
        "gold_kind",
        "pred_kind",
        "gold_tools",
        "pred_tools",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)
