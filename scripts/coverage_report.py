#!/usr/bin/env python3
"""Coverage report for data/final/dataset_final.jsonl.

Usage:
    python scripts/coverage_report.py
    python scripts/coverage_report.py --final data/final/dataset_final.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGETS: dict[str, int] = {
    "T1": 30,
    "T2": 20,
    "T3": 15,
    "T4": 15,
    "T5": 15,
    "T6": 25,
}

T6_STIMULUS_KEYWORDS = [
    ("cold", ["춥", "추워", "한랭", "cold"]),
    ("heat", ["더워", "덥", "heat"]),
    ("sleepy", ["졸", "피곤", "sleep"]),
    ("child_rear", ["아이", "애", "뒷좌석", "rear"]),
    ("visibility", ["안 보", "시야", "성에", "fog"]),
    ("noise", ["시끄", "소음", "noise"]),
    ("battery", ["배터리", "충전", "battery"]),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))
    return rows


def t2_combo_key(calls: list[dict[str, Any]]) -> str:
    tools = sorted(c.get("tool", "?") for c in calls)
    return " + ".join(tools) if tools else "(empty)"


def t5_pair_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    t5 = [r for r in records if r.get("utterance_type") == "T5"]
    by_utterance: dict[str, set[str]] = defaultdict(set)
    for r in t5:
        by_utterance[r.get("utterance", "")].add(r.get("expected", {}).get("kind", ""))

    complete = sum(1 for kinds in by_utterance.values() if "execute" in kinds and "confirm" in kinds)
    partial = len(by_utterance) - complete
    return {
        "utterances": len(by_utterance),
        "complete_pairs": complete,
        "incomplete": partial,
        "records": len(t5),
    }


def t6_stimulus_distribution(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for r in records:
        if r.get("utterance_type") != "T6":
            continue
        note = (r.get("note") or "") + " " + (r.get("utterance") or "")
        matched = False
        for axis, keywords in T6_STIMULUS_KEYWORDS:
            if any(kw in note for kw in keywords):
                counts[axis] += 1
                matched = True
        if not matched:
            counts["other"] += 1
    return counts


def labeled_by_ratio(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(r.get("labeled_by", "unknown") for r in records)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final dataset coverage report.")
    parser.add_argument("--final", type=Path, default=ROOT / "data" / "final" / "dataset_final.jsonl")
    args = parser.parse_args()

    final_path = args.final if args.final.is_absolute() else ROOT / args.final
    records = load_jsonl(final_path)

    print(f"Coverage report: {final_path}")
    print(f"Total records: {len(records)}\n")

    # Type counts vs targets
    type_counts = Counter(r.get("utterance_type") for r in records)
    type_rows: list[list[str]] = []
    for t in sorted(TARGETS.keys()):
        have = type_counts.get(t, 0)
        target = TARGETS[t]
        gap = target - have
        status = "OK" if gap <= 0 else f"-{gap}"
        type_rows.append([t, str(have), str(target), status])
    print("## 유형별 건수 (목표 대비)")
    print_table(["type", "have", "target", "gap"], type_rows)
    print()

    # T2 tool combos
    t2 = [r for r in records if r.get("utterance_type") == "T2"]
    combo_counts = Counter(t2_combo_key(r.get("expected", {}).get("calls", [])) for r in t2)
    print("## T2 툴 조합 분포")
    if combo_counts:
        combo_rows = [[combo, str(n)] for combo, n in combo_counts.most_common()]
        print_table(["combo", "count"], combo_rows)
    else:
        print("(no T2 records)")
    print()

    # T5 pairs
    t5 = t5_pair_stats(records)
    print("## T5 쌍 완성도")
    print(f"  unique utterances : {t5['utterances']}")
    print(f"  complete pairs    : {t5['complete_pairs']}")
    print(f"  incomplete        : {t5['incomplete']}")
    print(f"  total T5 records  : {t5['records']}")
    print()

    # T6 stimulus
    t6_dist = t6_stimulus_distribution(records)
    print("## T6 자극 축 (note+utterance 키워드)")
    if t6_dist:
        t6_rows = [[axis, str(n)] for axis, n in t6_dist.most_common()]
        print_table(["axis", "count"], t6_rows)
    else:
        print("(no T6 records)")
    print()

    # labeled_by
    lb = labeled_by_ratio(records)
    print("## labeled_by 비율")
    total = len(records) or 1
    lb_rows = [[name, str(n), f"{100 * n / total:.1f}%"] for name, n in lb.most_common()]
    print_table(["labeled_by", "count", "pct"], lb_rows)


if __name__ == "__main__":
    main()
