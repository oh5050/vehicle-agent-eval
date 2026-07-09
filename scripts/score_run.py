#!/usr/bin/env python3
"""Score a completed run: runs/{run_id}/logs.jsonl → reports/{run_id}/.

Usage:
    python scripts/score_run.py --run runs/e1_20260709_120000
    python scripts/score_run.py --run runs/e1_20260709_120000 --out reports/e1

Pipeline per inference entry: (parsed already in log) → registry.validate_call
→ normalizer → rule scorer → aggregator.

- Gold labels are read from the dataset referenced in the run manifest and are
  never modified.
- No retries: parse failures in the log are scored as parse_fail data.
- LLM judge is out of scope here (Stage 5).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aggregator import (  # noqa: E402
    by_label_source,
    confusion_to_csv,
    failure_class_counts,
    label_source_to_csv,
    latency_percentiles,
    latency_to_csv,
    matrix_to_csv,
    matrix_to_markdown,
    metrics_matrix,
    tool_confusion,
)
from src.arg_failures import arg_failures_to_csv, collect_arg_failures  # noqa: E402
from src.normalizer import load_synonyms  # noqa: E402
from src.registry import load_registry  # noqa: E402
from src.scorer import score_case  # noqa: E402

REQUIRED_ENTRY_KEYS = (
    "record_id",
    "utterance_type",
    "label_source",
    "parse_success",
    "parsed",
    "ttft_ms",
    "total_ms",
    "gate_enabled",
    "gate_applied",
    "consistency",
)


def load_log_entries(logs_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (manifest, inference entries) from logs.jsonl."""
    manifest: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    for line in logs_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        record = json.loads(s)
        entry_type = record.get("entry_type")
        if entry_type == "run_manifest" and manifest is None:
            manifest = record
        elif entry_type == "inference":
            entries.append(record)
    return manifest, entries


def validate_entries(entries: list[dict[str, Any]]) -> None:
    """Collect ALL missing required keys across entries and report at once."""
    problems: list[str] = []
    for i, entry in enumerate(entries):
        missing = [k for k in REQUIRED_ENTRY_KEYS if k not in entry]
        if missing:
            rid = entry.get("record_id", f"entry[{i}]")
            problems.append(f"{rid} (line {i}): missing {', '.join(missing)}")
    if problems:
        raise ValueError(
            "logs.jsonl inference entries missing required keys:\n  " + "\n  ".join(problems)
        )


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    """Gold dataset indexed by id (read-only)."""
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        record = json.loads(s)
        records[str(record.get("id"))] = record
    return records


def build_rows(
    entries: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    *,
    registry,
    synonyms: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_gold: list[str] = []

    for entry in entries:
        rid = str(entry["record_id"])
        gold = gold_by_id.get(rid)
        if gold is None:
            missing_gold.append(rid)
            continue

        expected = gold.get("expected") or {}
        parsed = entry.get("parsed")
        vehicle_state = gold.get("vehicle_state") or {}
        state_risk = gold.get("state_risk") or (
            "danger" if vehicle_state.get("speed_kmh", 0) > 0 else "safe"
        )
        score = score_case(
            expected,
            parsed,
            parse_success=bool(entry.get("parse_success")),
            registry=registry,
            synonyms=synonyms,
            order_sensitive=gold.get("order_sensitive"),
            gate_enabled=bool(entry.get("gate_enabled")),
            gate_applied=bool(entry.get("gate_applied")),
        )

        pred_calls = (parsed or {}).get("calls") or []
        rows.append(
            {
                "record_id": rid,
                "utterance_type": entry.get("utterance_type", gold.get("utterance_type", "?")),
                "label_source": entry.get("label_source", "unknown"),
                "gold_kind": expected.get("kind"),
                "pred_kind": (parsed or {}).get("kind"),
                "gold_tools": [c.get("tool", "") for c in (expected.get("calls") or [])],
                "pred_tools": [c.get("tool", "") for c in pred_calls],
                "consistency": entry.get("consistency"),
                "ttft_ms": entry.get("ttft_ms"),
                "total_ms": entry.get("total_ms"),
                "parse_success": bool(entry.get("parse_success")),
                "has_error": bool(entry.get("error") or entry.get("parse_error")),
                "order_sensitive": bool(gold.get("order_sensitive")),
                "gate_enabled": bool(entry.get("gate_enabled")),
                "state_risk": state_risk,
                "expected": expected,
                "parsed": parsed,
                "strict": score.strict,
                "normalized": score.normalized,
            }
        )

    if missing_gold:
        unique = sorted(set(missing_gold))
        raise ValueError(
            f"Gold records not found in dataset for {len(unique)} ids: {', '.join(unique[:10])}"
            + ("..." if len(unique) > 10 else "")
        )
    return rows


def _count_scoring_diffs(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (strict!=normalized count, normalized E-1 history count)."""
    diffs = 0
    e1_hist = 0
    for row in rows:
        strict = row["strict"]
        norm = row["normalized"]
        if (
            strict.failure_class != norm.failure_class
            or strict.arg_strict != norm.arg_strict
            or strict.correct != norm.correct
        ):
            diffs += 1
        if any(h.get("rule") == "E-1" for h in norm.normalization_history):
            e1_hist += 1
    return diffs, e1_hist


def write_parse_failures(entries: list[dict[str, Any]], path: Path) -> None:
    """Write parse_fail type distribution and raw_output samples."""
    failures = [e for e in entries if not e.get("parse_success")]
    by_type: dict[str, int] = {}
    for e in failures:
        t = e.get("utterance_type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    lines = [
        f"parse_fail total: {len(failures)} / {len(entries)}",
        "",
        "유형별 분포:",
    ]
    for t in sorted(by_type):
        lines.append(f"  {t}: {by_type[t]}")
    lines.extend(["", "raw_output samples:", ""])
    for e in failures:
        lines.append(f"--- {e.get('record_id')} | {e.get('utterance_type')} | {e.get('utterance', '')}")
        if e.get("error"):
            lines.append(f"error: {e['error']}")
        if e.get("parse_error"):
            lines.append(f"parse_error: {e['parse_error']}")
        raw = e.get("raw_output")
        if raw is None:
            lines.append("(no raw_output)")
        else:
            sample = raw if len(raw) <= 800 else raw[:800] + "..."
            lines.append(sample)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    run_id: str,
    entries: list[dict[str, Any]],
    registry,
    synonyms: dict[str, dict[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    n_inference = len(entries)
    n_parse_fail = sum(1 for e in entries if not e.get("parse_success"))
    n_error = sum(1 for e in entries if e.get("error") or e.get("parse_error"))
    n_gate_on = sum(1 for e in entries if e.get("gate_enabled"))
    n_scoring_diffs, n_e1_hist = _count_scoring_diffs(rows)

    matrices = {
        "strict": metrics_matrix(rows, variant="strict"),
        "normalized": metrics_matrix(rows, variant="normalized"),
    }
    for variant, matrix in matrices.items():
        matrix_to_csv(matrix, out_dir / f"metrics_{variant}.csv")

    confusion_to_csv(tool_confusion(rows, variant="normalized"), out_dir / "confusion_matrix.csv")
    latency_to_csv(latency_percentiles(rows), out_dir / "latency.csv")
    label_source_to_csv(
        by_label_source(rows, variant="normalized"), out_dir / "label_source_breakdown.csv"
    )

    arg_records = collect_arg_failures(
        rows, registry=registry, synonyms=synonyms, variant="normalized"
    )
    arg_failures_to_csv(arg_records, out_dir / "arg_failures.csv")
    write_parse_failures(entries, out_dir / "parse_failures.txt")
    arg_by_category = Counter(r["category"] for r in arg_records)

    fc_strict = failure_class_counts(rows, variant="strict")
    fc_norm = failure_class_counts(rows, variant="normalized")

    md_parts = [
        f"# Score report — {run_id}",
        "",
        f"[inference {n_inference} / parse_fail {n_parse_fail} / error {n_error}]",
        "",
        f"Rows scored: {len(rows)}",
        f"Gate enabled entries: {n_gate_on} / {n_inference}"
        + (" (gate_trigger_recall → N/A)" if n_gate_on == 0 else ""),
        f"strict≠normalized rows: {n_scoring_diffs} | E-1 normalization_history: {n_e1_hist}",
        "",
        f"wrong_args breakdown (normalized, n={len(arg_records)}): "
        + ", ".join(f"{k}={v}" for k, v in sorted(arg_by_category.items()))
        if arg_records
        else "wrong_args breakdown: (none)",
        "",
        matrix_to_markdown(matrices["strict"], title="Metrics (strict — 등가 규칙 적용 전)"),
        matrix_to_markdown(matrices["normalized"], title="Metrics (normalized — 등가 규칙 적용 후)"),
        "## Failure classes (strict / normalized)",
        "",
    ]
    all_classes = sorted(
        set(fc_strict["ALL(ref)"]) | set(fc_norm["ALL(ref)"])
    )
    md_parts.append("| failure_class | strict | normalized |")
    md_parts.append("|---|---|---|")
    for fc in all_classes:
        md_parts.append(
            f"| {fc} | {fc_strict['ALL(ref)'].get(fc, 0)} | {fc_norm['ALL(ref)'].get(fc, 0)} |"
        )
    md_parts.append("")
    md_parts.append(
        "주: false_execution_rate는 제1 지표이며 false_refusal_rate와 항상 쌍으로 볼 것. "
        "strict↔normalized 차이는 등가 규칙(E-1 등)의 기여분이다. "
        "T5-safe/danger는 state_risk 기준 분해. "
        "arg_failures.csv / parse_failures.txt 참고."
    )

    (out_dir / "summary.md").write_text("\n".join(md_parts), encoding="utf-8")


def score_run(
    run_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    dataset_path: Path | str | None = None,
) -> Path:
    """Score one run directory; returns the report output directory."""
    run_path = Path(run_dir)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    logs_path = run_path / "logs.jsonl"
    if not logs_path.exists():
        raise FileNotFoundError(f"logs.jsonl not found: {logs_path}")

    manifest, entries = load_log_entries(logs_path)
    if not entries:
        raise ValueError(f"No inference entries in {logs_path}")
    validate_entries(entries)

    run_id = (manifest or {}).get("run_id") or run_path.name

    if dataset_path is None:
        if manifest is None or "dataset_path" not in manifest:
            raise ValueError("Run manifest missing; pass dataset_path explicitly.")
        dataset_path = manifest["dataset_path"]
    ds_path = Path(dataset_path)
    if not ds_path.is_absolute():
        ds_path = ROOT / ds_path

    variant = (manifest or {}).get("registry_variant", "core8")
    registry = load_registry(variant)
    synonyms = load_synonyms()

    gold_by_id = load_dataset(ds_path)
    rows = build_rows(entries, gold_by_id, registry=registry, synonyms=synonyms)

    out = Path(out_dir) if out_dir is not None else ROOT / "reports" / run_id
    if not out.is_absolute():
        out = ROOT / out
    write_reports(rows, out, run_id=run_id, entries=entries, registry=registry, synonyms=synonyms)

    print(f"Scored {len(rows)} rows from {logs_path}")
    print(f"Reports → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a run's logs.jsonl and write reports.")
    parser.add_argument("--run", required=True, help="Run directory (e.g. runs/e1_20260709_120000)")
    parser.add_argument("--out", default=None, help="Output directory (default: reports/{run_id})")
    parser.add_argument("--dataset", default=None, help="Override gold dataset path")
    args = parser.parse_args()

    score_run(args.run, args.out, dataset_path=args.dataset)


if __name__ == "__main__":
    main()
