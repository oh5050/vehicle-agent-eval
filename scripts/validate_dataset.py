#!/usr/bin/env python3
"""Validate gold labels in data/final/ without modifying them.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --out violations.csv
    python scripts/validate_dataset.py --final data/final/dataset_final.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manual_add import (  # noqa: E402
    distractor_tool_names,
    normalize_utterance,
    validate_accept_calls,
    validate_calls_with_registry,
    validate_t3_item,
)
from src.registry import Registry, load_registry  # noqa: E402

DEFAULT_FINAL = ROOT / "data" / "final" / "dataset_final.jsonl"

NUMERIC_ARG_KEYS = frozenset({"temperature", "fan_level", "level", "volume", "target_pct"})

T6_COMMAND_PATTERNS = (
    "해줘",
    "켜줘",
    "꺼줘",
    "설정",
    "맞춰",
    "바꿔",
    "내려줘",
    "올려줘",
    "확인해",
)

T5_STATE_EXPOSURE_PATTERNS = (
    "주행 중",
    "정차",
    "멈춰",
    "서 있",
)


@dataclass(frozen=True)
class Violation:
    id: str
    utterance_type: str
    label_source: str
    violation_reason: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))
    return rows


def label_source(record: dict[str, Any]) -> str:
    if record.get("source_labeled_by") == "llm_draft":
        return "llm_draft"
    if record.get("labeled_by") in ("human_written", "llm_draft"):
        return str(record["labeled_by"])
    if record.get("labeled_by", "").startswith("llm"):
        return "llm_draft"
    return "human_written"


def _iter_call_lists(expected: dict[str, Any]) -> list[list[dict[str, Any]]]:
    calls = expected.get("calls") or []
    sequences = [calls] if calls else []
    for alt in expected.get("accept_calls") or []:
        if isinstance(alt, list):
            sequences.append(alt)
    return sequences


def _numeric_keys_in_calls(calls: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for call in calls:
        args = call.get("args") or {}
        for key in NUMERIC_ARG_KEYS:
            if key in args and key not in found:
                found.append(key)
    return found


def _utterance_has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _check_common(
    record: dict[str, Any],
    registry: Registry,
    distractor: set[str],
) -> list[str]:
    reasons: list[str] = []
    expected = record.get("expected") or {}
    kind = expected.get("kind", "")
    calls = expected.get("calls") or []

    if kind == "execute" and not calls:
        reasons.append("C1: kind=execute인데 calls가 비어 있음")

    if kind in ("refuse", "clarify") and calls:
        reasons.append(f"C2: kind={kind}인데 calls가 존재함")

    if calls:
        err = validate_calls_with_registry(registry, calls)
        if err:
            reasons.append(f"C3: {err}")

    accept_calls = expected.get("accept_calls", [])
    err = validate_accept_calls(registry, accept_calls, distractor)
    if err:
        reasons.append(f"C3: {err}")

    return reasons


def _check_t1_t2_numeric(record: dict[str, Any]) -> list[str]:
    utterance_type = record.get("utterance_type", "")
    if utterance_type not in ("T1", "T2"):
        return []

    utterance = record.get("utterance", "")
    expected = record.get("expected") or {}
    numeric_keys: list[str] = []

    for seq in _iter_call_lists(expected):
        numeric_keys.extend(_numeric_keys_in_calls(seq))

    if not numeric_keys:
        return []

    unique_keys = list(dict.fromkeys(numeric_keys))
    if not _utterance_has_digit(utterance):
        keys = ", ".join(unique_keys)
        return [f"T1/T2: 수치 인자({keys})가 있는데 발화에 숫자가 없음"]
    return []


def _check_t3(record: dict[str, Any], registry: Registry, distractor: set[str]) -> list[str]:
    if record.get("utterance_type") != "T3":
        return []
    expected = record.get("expected") or {}
    errors = validate_t3_item(record, expected, registry, distractor)
    return [f"T3: {e}" for e in errors]


def _check_t4(record: dict[str, Any]) -> list[str]:
    if record.get("utterance_type") != "T4":
        return []
    calls = (record.get("expected") or {}).get("calls") or []
    if calls:
        return ["T4: expected.calls가 비어 있어야 함"]
    return []


def _check_t5(record: dict[str, Any]) -> list[str]:
    if record.get("utterance_type") != "T5":
        return []

    utterance = record.get("utterance", "")

    if any(p in utterance for p in T5_STATE_EXPOSURE_PATTERNS):
        patterns = ", ".join(T5_STATE_EXPOSURE_PATTERNS)
        return [f"T5: 상태 노출 패턴 감지 ({patterns})"]

    return []


def _check_t6(record: dict[str, Any]) -> list[str]:
    if record.get("utterance_type") != "T6":
        return []
    utterance = record.get("utterance", "")
    matched = [p for p in T6_COMMAND_PATTERNS if p in utterance]
    if matched:
        return [f"T6: 직접 명령 패턴 감지 ({', '.join(matched)})"]
    return []


def validate_record(
    record: dict[str, Any],
    *,
    registry: Registry,
    distractor: set[str],
) -> list[Violation]:
    rid = str(record.get("id", "?"))
    utype = str(record.get("utterance_type", "?"))
    src = label_source(record)

    reasons: list[str] = []
    reasons.extend(_check_common(record, registry, distractor))
    reasons.extend(_check_t1_t2_numeric(record))
    reasons.extend(_check_t3(record, registry, distractor))
    reasons.extend(_check_t4(record))
    reasons.extend(_check_t5(record))
    reasons.extend(_check_t6(record))

    return [
        Violation(id=rid, utterance_type=utype, label_source=src, violation_reason=reason)
        for reason in reasons
    ]


def find_duplicate_utterances(records: list[dict[str, Any]]) -> list[Violation]:
    """Detect normalized utterance duplicates (T5 excluded — same utterance × states is intentional)."""
    norm_to_entries: dict[str, list[tuple[str, str, str]]] = {}
    for record in records:
        if record.get("utterance_type") == "T5":
            continue
        utterance = record.get("utterance", "")
        norm = normalize_utterance(utterance)
        if not norm:
            continue
        rid = str(record.get("id", "?"))
        utype = str(record.get("utterance_type", "?"))
        src = label_source(record)
        norm_to_entries.setdefault(norm, []).append((rid, utype, src))

    violations: list[Violation] = []
    for entries in norm_to_entries.values():
        if len(entries) < 2:
            continue
        all_ids = [e[0] for e in entries]
        for rid, utype, src in entries:
            others = [i for i in all_ids if i != rid]
            violations.append(
                Violation(
                    id=rid,
                    utterance_type=utype,
                    label_source=src,
                    violation_reason=f"C4: 발화 정규화 후 중복 (동일: {', '.join(others)})",
                )
            )
    return violations


def validate_dataset(
    records: list[dict[str, Any]],
    *,
    registry: Registry | None = None,
) -> list[Violation]:
    reg = registry or load_registry("core8")
    distractor = distractor_tool_names()

    violations: list[Violation] = []
    for record in records:
        violations.extend(validate_record(record, registry=reg, distractor=distractor))
    violations.extend(find_duplicate_utterances(records))
    return violations


def write_csv(path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "utterance_type", "label_source", "violation_reason"],
        )
        writer.writeheader()
        for v in violations:
            writer.writerow(
                {
                    "id": v.id,
                    "utterance_type": v.utterance_type,
                    "label_source": v.label_source,
                    "violation_reason": v.violation_reason,
                }
            )


def print_violations(violations: list[Violation]) -> None:
    if not violations:
        print("No violations.")
        return

    headers = ("id", "utterance_type", "label_source", "violation_reason")
    widths = [len(h) for h in headers]
    rows = [
        (v.id, v.utterance_type, v.label_source, v.violation_reason)
        for v in violations
    ]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data/final gold labels (read-only).")
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL, help="Path to final JSONL")
    parser.add_argument("--out", type=Path, default=None, help="Write violations to CSV")
    args = parser.parse_args()

    final_path = args.final if args.final.is_absolute() else ROOT / args.final
    records = load_jsonl(final_path)
    violations = validate_dataset(records)

    print(f"Validated {len(records)} records from {final_path}")
    print(f"Violations: {len(violations)}")
    print_violations(violations)

    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        write_csv(out_path, violations)
        print(f"\nWrote {len(violations)} rows → {out_path}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
