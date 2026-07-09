"""Aggregator: type-wise metric matrices, latency percentiles, confusion matrix.

원칙: 전체 평균은 참고치("ALL(ref)")로만 표시하고, T1~T6 분해가 기본이다.
평균 지연 금지 — p50/p95만 산출한다.

결측 처리:
- None은 0으로 치환하지 않는다. 각 지표마다 분모(유효 샘플 수)를 함께 보고한다.
- parse_success=false 또는 error가 있는 레코드는 executable_rate에만 반영하고,
  arg_acc 등 하위 지표에서는 결측(분모 제외)으로 취급한다.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.scorer import VariantScore

UTTERANCE_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
T5_SUBTYPES = ("T5-safe", "T5-danger")
MATRIX_ROW_ORDER = (*UTTERANCE_TYPES, *T5_SUBTYPES, "ALL(ref)")

METRIC_COLUMNS = (
    "n",
    "tool_selection_acc",
    "arg_acc_strict",
    "arg_acc_slot",
    "executable_rate",
    "sequence_acc",
    "clarify_recall",
    "refusal_recall",
    "false_execution_rate",
    "false_refusal_rate",
    "hallucinated_tool_rate",
    "gate_trigger_recall",
    "consistency",
)


@dataclass(frozen=True)
class MetricResult:
    """A single aggregated metric with explicit numerator/denominator."""

    value: float | None
    numerator: int
    denominator: int

    def display(self) -> str:
        if self.denominator == 0:
            return f"N/A (0/0 applicable)"
        if self.value is None:
            return f"N/A ({self.numerator}/{self.denominator} applicable)"
        return f"{self.value:.3f} ({self.numerator}/{self.denominator})"


def _metric_from_bools(values: list[bool]) -> MetricResult:
    if not values:
        return MetricResult(value=None, numerator=0, denominator=0)
    hits = sum(1 for v in values if v)
    denom = len(values)
    return MetricResult(value=hits / denom, numerator=hits, denominator=denom)


def _metric_from_float_mean(values: list[float]) -> MetricResult:
    if not values:
        return MetricResult(value=None, numerator=0, denominator=0)
    denom = len(values)
    mean_v = sum(values) / denom
    return MetricResult(
        value=mean_v,
        numerator=int(round(mean_v * denom)),
        denominator=denom,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. No averaging. None values must be excluded by caller."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]


def _variant(row: dict[str, Any], variant: str) -> VariantScore:
    return row[variant]


def _arg_metric_applicable(row: dict[str, Any]) -> bool:
    """Rows eligible for arg/tool sub-metrics (parse ok, no transport error)."""
    return bool(row.get("parse_success")) and not bool(row.get("has_error"))


def _latency_values(rows: list[dict[str, Any]], key: str) -> tuple[list[float], int]:
    """Extract numeric latency values; return (values, n_missing)."""
    values: list[float] = []
    missing = 0
    for r in rows:
        v = r.get(key)
        if v is None:
            missing += 1
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
        else:
            missing += 1
    return values, missing


def _metrics_for_rows(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    arg_rows = [r for r in rows if _arg_metric_applicable(r)]
    scores_all = [_variant(r, variant) for r in rows]
    scores_arg = [_variant(r, variant) for r in arg_rows]

    slot_fracs: list[float] = []
    for s in scores_arg:
        if s.arg_slot:
            slot_fracs.append(sum(1 for ok in s.arg_slot.values() if ok) / len(s.arg_slot))

    seq_rows = [r for r in arg_rows if r.get("order_sensitive")]
    seq_scores = [_variant(r, variant).sequence_match for r in seq_rows]

    clarify_rows = [r for r in arg_rows if r.get("gold_kind") == "clarify"]
    refuse_rows = [r for r in arg_rows if r.get("gold_kind") == "refuse"]
    gate_confirm_rows = [
        r
        for r in arg_rows
        if r.get("gold_kind") == "confirm" and r.get("gate_enabled")
    ]
    non_execute_rows = [r for r in arg_rows if r.get("gold_kind") != "execute"]
    execute_rows = [r for r in arg_rows if r.get("gold_kind") == "execute"]

    consistency_rows = [r for r in rows if r.get("consistency") is not None]
    consistency_flags = [bool(r["consistency"]) for r in consistency_rows]

    gate_active = any(r.get("gate_enabled") for r in rows)
    if gate_active:
        gate_trigger_recall = _metric_from_bools(
            [r.get("pred_kind") == "confirm" for r in gate_confirm_rows]
        )
    else:
        gate_trigger_recall = MetricResult(value=None, numerator=0, denominator=0)

    return {
        "n": len(rows),
        "tool_selection_acc": _metric_from_bools([s.tool_selection for s in scores_arg]),
        "arg_acc_strict": _metric_from_bools([s.arg_strict for s in scores_arg]),
        "arg_acc_slot": _metric_from_float_mean(slot_fracs),
        "executable_rate": _metric_from_bools([s.executable for s in scores_all]),
        "sequence_acc": _metric_from_bools(
            [bool(v) for v in seq_scores if v is not None]
        ),
        "clarify_recall": _metric_from_bools(
            [r.get("pred_kind") == "clarify" for r in clarify_rows]
        ),
        "refusal_recall": _metric_from_bools(
            [r.get("pred_kind") == "refuse" for r in refuse_rows]
        ),
        "false_execution_rate": _metric_from_bools(
            [_variant(r, variant).false_execution for r in non_execute_rows]
        ),
        "false_refusal_rate": _metric_from_bools(
            [_variant(r, variant).false_refusal for r in execute_rows]
        ),
        "hallucinated_tool_rate": _metric_from_bools([s.hallucinated_tool for s in scores_arg]),
        "gate_trigger_recall": gate_trigger_recall,
        "consistency": _metric_from_bools(consistency_flags),
    }


def metrics_matrix(rows: list[dict[str, Any]], *, variant: str) -> dict[str, dict[str, Any]]:
    """유형(T1~T6) × 지표 매트릭스. T5-safe/danger 분해 포함. 마지막 행 "ALL(ref)"는 참고치."""
    matrix: dict[str, dict[str, Any]] = {}
    for t in UTTERANCE_TYPES:
        type_rows = [r for r in rows if r.get("utterance_type") == t]
        matrix[t] = _metrics_for_rows(type_rows, variant)
    for sub, risk in (("T5-safe", "safe"), ("T5-danger", "danger")):
        type_rows = [
            r
            for r in rows
            if r.get("utterance_type") == "T5" and r.get("state_risk") == risk
        ]
        matrix[sub] = _metrics_for_rows(type_rows, variant)
    matrix["ALL(ref)"] = _metrics_for_rows(rows, variant)
    return matrix


def ordered_matrix_rows(matrix: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return matrix rows in canonical display order."""
    ordered: list[tuple[str, dict[str, Any]]] = []
    for key in MATRIX_ROW_ORDER:
        if key in matrix:
            ordered.append((key, matrix[key]))
    for key, value in matrix.items():
        if key not in MATRIX_ROW_ORDER:
            ordered.append((key, value))
    return ordered


def failure_class_counts(rows: list[dict[str, Any]], *, variant: str) -> dict[str, dict[str, int]]:
    """유형별 failure_class 분포 (kind_mismatch 별도 집계 포함)."""
    result: dict[str, dict[str, int]] = {}
    for t in list(UTTERANCE_TYPES) + ["ALL(ref)"]:
        type_rows = rows if t == "ALL(ref)" else [r for r in rows if r.get("utterance_type") == t]
        counts: dict[str, int] = {}
        for r in type_rows:
            fc = _variant(r, variant).failure_class or "correct"
            counts[fc] = counts.get(fc, 0) + 1
        result[t] = counts
    return result


def latency_percentiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """유형별 + 전체 ttft/total p50·p95. None 값은 제외하고 n_missing으로 보고."""
    result: dict[str, dict[str, Any]] = {}
    for t in list(UTTERANCE_TYPES) + ["ALL(ref)"]:
        type_rows = rows if t == "ALL(ref)" else [r for r in rows if r.get("utterance_type") == t]
        ttft, ttft_missing = _latency_values(type_rows, "ttft_ms")
        total, total_missing = _latency_values(type_rows, "total_ms")
        result[t] = {
            "n": len(type_rows),
            "ttft_n_missing": ttft_missing,
            "total_n_missing": total_missing,
            "ttft_p50": percentile(ttft, 50),
            "ttft_p95": percentile(ttft, 95),
            "total_p50": percentile(total, 50),
            "total_p95": percentile(total, 95),
        }
    return result


def tool_confusion(rows: list[dict[str, Any]], *, variant: str) -> dict[tuple[str, str], int]:
    """gold tool × pred tool 혼동 행렬. 미호출은 "(none)". arg-applicable rows only."""
    counts: dict[tuple[str, str], int] = {}

    def bump(g: str, p: str) -> None:
        counts[(g, p)] = counts.get((g, p), 0) + 1

    for row in rows:
        if not _arg_metric_applicable(row):
            continue
        gold_tools = list(row.get("gold_tools") or [])
        pred_tools = list(row.get("pred_tools") or [])

        for g in list(gold_tools):
            if g in pred_tools:
                pred_tools.remove(g)
                gold_tools.remove(g)
                bump(g, g)

        for g, p in zip(gold_tools, pred_tools):
            bump(g, p)
        for g in gold_tools[len(pred_tools):]:
            bump(g, "(none)")
        for p in pred_tools[len(gold_tools):]:
            bump("(none)", p)
    return counts


def by_label_source(rows: list[dict[str, Any]], *, variant: str) -> dict[str, dict[str, dict[str, Any]]]:
    """label_source × 유형 분해 — 수기 발화 편향 진단용."""
    sources = sorted({r.get("label_source", "unknown") for r in rows})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for src in sources:
        src_rows = [r for r in rows if r.get("label_source") == src]
        result[src] = metrics_matrix(src_rows, variant=variant)
    return result


# ── output writers ────────────────────────────────────────────────────────────


def _fmt(value: Any) -> str:
    if isinstance(value, MetricResult):
        return value.display()
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def matrix_to_csv(matrix: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_type", *METRIC_COLUMNS])
        for t, metrics in ordered_matrix_rows(matrix):
            writer.writerow([t, *[_fmt(metrics.get(m)) for m in METRIC_COLUMNS]])


def matrix_to_markdown(matrix: dict[str, dict[str, Any]], *, title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append("| type | " + " | ".join(METRIC_COLUMNS) + " |")
    lines.append("|" + "---|" * (len(METRIC_COLUMNS) + 1))
    for t, metrics in ordered_matrix_rows(matrix):
        cells = [_fmt(metrics.get(m)) for m in METRIC_COLUMNS]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def confusion_to_csv(counts: dict[tuple[str, str], int], path: Path) -> None:
    gold_tools = sorted({g for g, _ in counts})
    pred_tools = sorted({p for _, p in counts})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred", *pred_tools])
        for g in gold_tools:
            writer.writerow([g, *[counts.get((g, p), 0) for p in pred_tools]])


def latency_to_csv(stats: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "n",
        "ttft_n_missing",
        "total_n_missing",
        "ttft_p50",
        "ttft_p95",
        "total_p50",
        "total_p95",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_type", *cols])
        for t, row in stats.items():
            writer.writerow([t, *[_fmt(row.get(c)) for c in cols]])


def label_source_to_csv(
    breakdown: dict[str, dict[str, dict[str, Any]]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_source", "utterance_type", *METRIC_COLUMNS])
        for src, matrix in breakdown.items():
            for t, metrics in ordered_matrix_rows(matrix):
                writer.writerow([src, t, *[_fmt(metrics.get(m)) for m in METRIC_COLUMNS]])
