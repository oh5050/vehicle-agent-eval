#!/usr/bin/env python3
"""Diagnose strict vs normalized scoring paths for a completed run.

Usage:
    python scripts/diagnose_scoring.py --run runs/e1_20260709_150330
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_run import build_rows, load_dataset, load_log_entries  # noqa: E402
from src.normalizer import load_synonyms, normalize_calls  # noqa: E402
from src.registry import load_registry  # noqa: E402
from src.scorer import score_case  # noqa: E402

WINDOW_PARTS = frozenset({"driver", "passenger", "rear_left", "rear_right"})


def _window_control_count(calls: list[dict]) -> int:
    return sum(1 for c in calls if c.get("tool") == "window_control")


def _is_four_window_pred(calls: list[dict]) -> bool:
    wc = [c for c in calls if c.get("tool") == "window_control"]
    if len(wc) != 4:
        return False
    windows = {c.get("args", {}).get("window") for c in wc}
    return windows == set(WINDOW_PARTS)


def diagnose_run(run_dir: Path, *, dataset_path: Path | None = None) -> None:
    logs_path = run_dir / "logs.jsonl"
    manifest, entries = load_log_entries(logs_path)

    if dataset_path is None:
        if manifest is None or "dataset_path" not in manifest:
            raise ValueError("dataset_path required when manifest is missing")
        dataset_path = ROOT / manifest["dataset_path"]
    elif not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    variant = (manifest or {}).get("registry_variant", "core8")
    registry = load_registry(variant)
    synonyms = load_synonyms()
    gold_by_id = load_dataset(dataset_path)
    rows = build_rows(entries, gold_by_id, registry=registry, synonyms=synonyms)

    diff_rows: list[dict] = []
    e1_rows: list[dict] = []
    e1_no_diff: list[dict] = []
    four_window_preds: list[dict] = []

    for entry, row in zip(entries, rows):
        rid = entry["record_id"]
        gold = gold_by_id[rid]
        parsed = entry.get("parsed")
        score = score_case(
            gold["expected"],
            parsed,
            parse_success=bool(entry.get("parse_success")),
            registry=registry,
            synonyms=synonyms,
            order_sensitive=gold.get("order_sensitive"),
            gate_enabled=bool(entry.get("gate_enabled")),
            gate_applied=bool(entry.get("gate_applied")),
        )

        strict = score.strict
        norm = score.normalized
        changed = (
            strict.failure_class != norm.failure_class
            or strict.arg_strict != norm.arg_strict
            or strict.correct != norm.correct
        )
        if changed:
            diff_rows.append(
                {
                    "record_id": rid,
                    "strict_fc": strict.failure_class,
                    "norm_fc": norm.failure_class,
                    "strict_arg": strict.arg_strict,
                    "norm_arg": norm.arg_strict,
                }
            )

        has_e1 = any(h.get("rule") == "E-1" for h in norm.normalization_history)
        if has_e1:
            e1_rows.append({"record_id": rid, "changed": changed})
            if not changed:
                e1_no_diff.append({"record_id": rid, "fc": norm.failure_class})

        if entry.get("parse_success") and parsed:
            calls = parsed.get("calls") or []
            if _is_four_window_pred(calls):
                four_window_preds.append(
                    {
                        "record_id": rid,
                        "gold_calls": len(gold["expected"].get("calls") or []),
                        "strict_fc": strict.failure_class,
                        "norm_fc": norm.failure_class,
                        "changed": changed,
                    }
                )

    print(f"# Scoring diagnostic - {run_dir.name}")
    print()
    print("## strict vs normalized paths")
    print(f"- score_case calls _score_variant(equivalence=False) for strict")
    print(f"- score_case calls _score_variant(equivalence=True) for normalized")
    print(f"- rows scored: {len(rows)}")
    print(f"- rows where strict != normalized: {len(diff_rows)}")
    if diff_rows[:5]:
        print("  sample diffs:")
        for d in diff_rows[:5]:
            print(f"    {d}")
    print()

    print("## E-1 (window_control 4-call collapse)")
    print(f"- normalization_history with E-1: {len(e1_rows)}")
    print(f"- E-1 fired but score unchanged: {len(e1_no_diff)}")
    if e1_no_diff[:5]:
        print("  (E-1 alone did not flip failure_class — e.g. gold has empty calls or other mismatch)")
        for d in e1_no_diff[:5]:
            print(f"    {d}")
    print(f"- predictions with 4 per-window window_control calls: {len(four_window_preds)}")
    for d in four_window_preds[:5]:
        print(f"    {d}")

    # Direct E-1 unit probe on score_run path
    probe_calls = [
        {"tool": "window_control", "args": {"window": w, "action": "close"}}
        for w in sorted(WINDOW_PARTS)
    ]
    probe = normalize_calls(probe_calls, registry=registry, synonyms=synonyms, equivalence=True)
    print(f"- direct normalize_calls(E-1 probe) history rules: {[h.get('rule') for h in probe.history]}")
    print()

    print("## T5 vehicle_state / state_risk distribution (inference entries)")
    t5_entries = [e for e in entries if e.get("utterance_type") == "T5"]
    risk_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    speed_counter: Counter[str] = Counter()
    for e in t5_entries:
        g = gold_by_id[e["record_id"]]
        risk = g.get("state_risk") or (
            "danger" if g["vehicle_state"].get("speed_kmh", 0) > 0 else "safe"
        )
        risk_counter[risk] += 1
        kind_counter[g["expected"]["kind"]] += 1
        speed_counter[str(g["vehicle_state"].get("speed_kmh"))] += 1
    print(f"- T5 inference rows: {len(t5_entries)}")
    print(f"- state_risk: {dict(risk_counter)}")
    print(f"- gold kind: {dict(kind_counter)}")
    print(f"- speed_kmh: {dict(speed_counter)}")
    print()

    print("## Gate")
    gate_on = sum(1 for e in entries if e.get("gate_enabled"))
    print(f"- gate_enabled=true entries: {gate_on} / {len(entries)}")
    if gate_on == 0:
        print("- gate_trigger_recall should be N/A (not 0.0) for this run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose strict/normalized scoring for a run")
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--dataset", default=None, help="Override gold dataset path")
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    ds = Path(args.dataset) if args.dataset else None
    diagnose_run(run_path, dataset_path=ds)


if __name__ == "__main__":
    main()
