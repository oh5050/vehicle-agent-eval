#!/usr/bin/env python3
"""Add human-written records to data/final/ — interactive, template, or batch import.

Usage:
    python scripts/manual_add.py
    python scripts/manual_add.py --template T2 --count 12
    python scripts/manual_add.py --import data/drafts/manual_template_T2.yaml
    python scripts/manual_add.py --import data/drafts/manual_template_T2.yaml --replace
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_schema import VALID_EXPECTED_KINDS, Expected, UtteranceType  # noqa: E402
from src.label_schema import FinalLabelRecord, LabelHistoryEntry, utc_now_iso  # noqa: E402
from src.registry import Registry, load_registry  # noqa: E402

UTTERANCE_TYPES: tuple[UtteranceType, ...] = ("T1", "T2", "T3", "T4", "T5", "T6")

VEHICLE_PRESETS: dict[str, dict[str, Any]] = {
    "safe": {
        "speed_kmh": 0,
        "gear": "P",
        "battery_pct": 72,
        "outside_temp_c": 3,
        "passengers": {"front": True, "rear": False},
    },
    "driving": {
        "speed_kmh": 60,
        "gear": "D",
        "battery_pct": 72,
        "outside_temp_c": 8,
        "passengers": {"front": True, "rear": False},
    },
    "cold": {
        "speed_kmh": 0,
        "gear": "P",
        "battery_pct": 65,
        "outside_temp_c": -8,
        "passengers": {"front": True, "rear": False},
    },
    "night": {
        "speed_kmh": 40,
        "gear": "D",
        "battery_pct": 55,
        "outside_temp_c": 2,
        "passengers": {"front": True, "rear": True},
    },
}

T5_OPPOSITE_KIND = {"execute": "confirm", "confirm": "execute"}

T6_NUMERIC_WARN_KEYS = frozenset({"temperature", "volume", "level", "fan_level"})

TEMPLATE_HEADER_RULES: dict[str, list[str]] = {
    "T2": [
        "T2 = 서로 다른 2개 이상 툴, 수치 인자는 발화에 숫자로 명시,",
        '     order_sensitive는 "앞 호출 결과가 뒤 호출 인자를 결정하는가"로 판정',
    ],
    "T3": [
        "T3 = kind=clarify, calls=[], intended_tool에 의도 툴 명시,",
        "     missing_slots는 intended_tool 스키마의 required/required_one_of 슬롯만 사용,",
        "     required_one_of 툴(climate_set 등)은 missing_slots가 해당 그룹 전체를 포함해야 함",
    ],
    "T6": [
        "T6 = 직접 명령 금지, 상태·불편만 표현, 수치 인자 필요한 툴 회피,",
        "     복수 해석 자연스러우면 accept_also에 양쪽",
    ],
}


@dataclass
class ImportItemResult:
    item_id: str
    ok: bool
    skipped: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record: FinalLabelRecord | None = None


@dataclass
class ImportSummary:
    added: int = 0
    skipped: int = 0
    failed: int = 0
    replaced: int = 0


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        records.append(json.loads(stripped))
    return records


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_utterance(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[\s]+", "", t)
    t = re.sub(r"[.,!?~'\"…·\-]", "", t)
    return t


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def find_utterance_conflicts(utterance: str, existing: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    norm_new = normalize_utterance(utterance)
    for row in existing:
        other = row.get("utterance", "")
        if utterance == other:
            warnings.append(f"완전 일치: id={row.get('id')}")
            continue
        norm_other = normalize_utterance(other)
        if norm_other and levenshtein(norm_new, norm_other) <= 3:
            warnings.append(f"유사 발화(편집거리≤3): id={row.get('id')} → {other!r}")
    return warnings


def validate_t2_calls(calls: list[dict[str, Any]]) -> str | None:
    if len(calls) < 2:
        return "T2는 calls가 2개 이상이어야 합니다."
    tools = [c.get("tool") for c in calls]
    if any(not t for t in tools):
        return "각 call에 tool 키가 필요합니다."
    if len(set(tools)) != len(tools):
        return "T2 calls의 tool 이름은 서로 달라야 합니다."
    return None


def validate_t3_item(
    item: dict[str, Any],
    expected_raw: dict[str, Any],
    registry: Registry,
    distractor: set[str],
) -> list[str]:
    """Return T3 validation errors (empty list = pass)."""
    errors: list[str] = []
    kind = expected_raw.get("kind", "")
    calls = expected_raw.get("calls") or []
    missing_slots = expected_raw.get("missing_slots", [])
    intended_tool = (item.get("intended_tool") or "").strip()

    if kind != "clarify":
        errors.append(f"T3는 kind='clarify'여야 합니다 (현재: {kind!r}).")

    if kind == "clarify" and calls:
        errors.append("T3 clarify는 expected.calls가 비어 있어야 합니다.")

    if not intended_tool:
        errors.append("T3는 intended_tool 필드가 필요합니다.")
        return errors

    if intended_tool in distractor:
        errors.append(f"distractor 툴 금지: {intended_tool}")
        return errors

    tool = registry.get_tool(intended_tool)
    if tool is None:
        errors.append(f"intended_tool {intended_tool!r}이(가) registry에 없습니다.")
        return errors

    if not isinstance(missing_slots, list):
        errors.append("expected.missing_slots는 배열이어야 합니다.")
        return errors

    if kind == "clarify" and not missing_slots:
        errors.append("T3 clarify는 missing_slots가 비어 있으면 안 됩니다.")

    allowed = set(tool.required)
    if tool.required_one_of:
        allowed |= set(tool.required_one_of)

    for slot in missing_slots:
        if not isinstance(slot, str):
            errors.append(f"missing_slots 항목은 문자열이어야 합니다: {slot!r}")
            continue
        if slot not in allowed:
            errors.append(
                f"missing_slots {slot!r}은(는) {intended_tool}의 "
                f"required/required_one_of에 없습니다."
            )

    if tool.required_one_of:
        one_of_set = set(tool.required_one_of)
        missing_set = set(missing_slots)
        if not one_of_set.issubset(missing_set):
            errors.append(
                f"{intended_tool}의 required_one_of {list(tool.required_one_of)} 전체가 "
                f"missing_slots에 포함되어야 합니다 (현재: {missing_slots}). "
                f"일부만 누락된 경우는 clarify가 아닙니다."
            )

    return errors


def validate_calls_with_registry(registry: Registry, calls: list[dict[str, Any]]) -> str | None:
    for call in calls:
        tool = call.get("tool")
        if not tool:
            return "call에 tool이 없습니다."
        args = call.get("args", {})
        if not isinstance(args, dict):
            return f"{tool}: args는 객체여야 합니다."
        result = registry.validate_call(tool, args)
        if not result.valid:
            return f"{tool}: [{result.error_code}] {result.error_message}"
    return None


def validate_accept_calls(
    registry: Registry,
    accept_calls: Any,
    distractor: set[str],
) -> str | None:
    if not isinstance(accept_calls, list):
        return "expected.accept_calls는 배열이어야 합니다."
    for i, seq in enumerate(accept_calls):
        if not isinstance(seq, list):
            return f"accept_calls[{i}]는 call 배열이어야 합니다."
        err = calls_use_distractor(seq, distractor)
        if err:
            return f"accept_calls[{i}]: {err}"
        err = validate_calls_with_registry(registry, seq)
        if err:
            return f"accept_calls[{i}]: {err}"
    return None


def distractor_tool_names() -> set[str]:
    core8 = set(load_registry("core8").tool_names)
    full18 = set(load_registry("full18").tool_names)
    return full18 - core8


def calls_use_distractor(calls: list[dict[str, Any]], distractor: set[str]) -> str | None:
    for call in calls:
        tool = call.get("tool")
        if tool in distractor:
            return f"distractor 툴 금지: {tool}"
    return None


def t6_numeric_arg_warning(calls: list[dict[str, Any]]) -> str | None:
    for call in calls:
        args = call.get("args") or {}
        for key in T6_NUMERIC_WARN_KEYS:
            if key in args:
                return f"T6에 수치형 인자 {key}={args[key]!r} — 회피 권장"
    return None


def count_by_type(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {t: 0 for t in UTTERANCE_TYPES}
    for row in records:
        t = row.get("utterance_type")
        if t in counts:
            counts[t] += 1
    return counts


def next_manual_id(existing: list[dict[str, Any]], *, prefix: str = "manual") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n = sum(1 for r in existing if str(r.get("id", "")).startswith(f"{prefix}_"))
    return f"{prefix}_{stamp}_{n + 1:03d}"


def template_item(utterance_type: str, index: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": f"manual_{utterance_type}_{index:03d}",
        "utterance": "",
        "expected": {
            "kind": "",
            "calls": [],
            "accept_also": [],
            "accept_calls": [],
            "missing_slots": [],
            "order_sensitive": None,
        },
        "vehicle_state_preset": "",
        "note": "",
    }
    if utterance_type == "T3":
        item["intended_tool"] = ""
    return item


def build_template_header(utterance_type: str) -> str:
    lines = [
        "# Human-written batch template — fill utterance and expected fields.",
        "# Gold registry variant: core8 only (distractor tools forbidden).",
        "",
    ]
    rules = TEMPLATE_HEADER_RULES.get(utterance_type, [])
    if rules:
        lines.append(f"# {rules[0]}")
        for rule in rules[1:]:
            lines.append(f"# {rule}")
        lines.append("")
    return "\n".join(lines)


def generate_template(path: Path, utterance_type: str, count: int) -> None:
    if utterance_type not in UTTERANCE_TYPES:
        raise ValueError(f"Unknown utterance_type: {utterance_type}")

    payload = {
        "utterance_type": utterance_type,
        "records": [template_item(utterance_type, i + 1) for i in range(count)],
    }
    header = build_template_header(utterance_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(header + body, encoding="utf-8")
    print(f"Wrote template ({count} items) → {path}")


def yaml_item_to_record(
    item: dict[str, Any],
    *,
    utterance_type: str,
    author: str,
    existing: list[dict[str, Any]],
) -> FinalLabelRecord:
    expected_raw = item.get("expected", {})
    calls = expected_raw.get("calls", [])
    expected = Expected(
        kind=expected_raw.get("kind", "execute"),
        calls=calls,
        accept_also=expected_raw.get("accept_also", []),
        accept_calls=expected_raw.get("accept_calls", []),
        reason_keywords=expected_raw.get("reason_keywords", []),
        missing_slots=expected_raw.get("missing_slots", []),
    )
    preset = (item.get("vehicle_state_preset") or "safe").lower()
    if preset not in VEHICLE_PRESETS:
        raise ValueError(f"unknown vehicle_state_preset: {preset!r}")

    order_sensitive = expected_raw.get("order_sensitive")
    if order_sensitive is None:
        order_sensitive = item.get("order_sensitive")

    state_risk = item.get("state_risk")
    if utterance_type == "T5" and state_risk is None:
        state_risk = "safe" if preset == "safe" else "danger" if preset == "driving" else None

    item_id = item.get("id") or next_manual_id(existing)
    note = item.get("note") or ""
    intended_tool = (item.get("intended_tool") or "").strip() or None

    return FinalLabelRecord(
        id=item_id,
        utterance_type=utterance_type,  # type: ignore[arg-type]
        utterance=item["utterance"],
        vehicle_state=json.loads(json.dumps(VEHICLE_PRESETS[preset])),
        expected=expected,
        labeled_by="human_written",
        label_history=[
            LabelHistoryEntry(
                at=utc_now_iso(),
                action="manual_add",
                reviewer=author,
                note=note or None,
                utterance_type=utterance_type,  # type: ignore[arg-type]
                expected=expected.model_copy(deep=True),
            )
        ],
        order_sensitive=order_sensitive,
        state_risk=state_risk,
        intended_tool=intended_tool,
    )


def validate_import_item(
    item: dict[str, Any],
    *,
    utterance_type: str,
    registry: Registry,
    existing: list[dict[str, Any]],
    distractor: set[str],
    batch_utterances: list[str],
) -> ImportItemResult:
    item_id = str(item.get("id", "?"))
    errors: list[str] = []
    warnings: list[str] = []

    utterance = (item.get("utterance") or "").strip()
    if not utterance:
        errors.append("utterance가 비어 있습니다.")
        return ImportItemResult(item_id=item_id, ok=False, errors=errors)

    expected_raw = item.get("expected") or {}
    kind = expected_raw.get("kind", "")
    if kind not in VALID_EXPECTED_KINDS:
        errors.append(f"유효하지 않은 kind: {kind!r}")
        return ImportItemResult(item_id=item_id, ok=False, errors=errors)

    calls = expected_raw.get("calls") or []
    if not isinstance(calls, list):
        errors.append("expected.calls는 배열이어야 합니다.")
        return ImportItemResult(item_id=item_id, ok=False, errors=errors)

    if utterance_type == "T4":
        if calls:
            errors.append("T4(refuse)는 expected.calls가 비어 있어야 합니다.")
        if kind != "refuse":
            warnings.append(f"T4인데 kind={kind!r} — refuse 권장")

    if utterance_type == "T2":
        err = validate_t2_calls(calls)
        if err:
            errors.append(err)

    if utterance_type == "T3":
        errors.extend(validate_t3_item(item, expected_raw, registry, distractor))

    if calls:
        err = calls_use_distractor(calls, distractor)
        if err:
            errors.append(err)
        err = validate_calls_with_registry(registry, calls)
        if err:
            errors.append(err)

    accept_calls = expected_raw.get("accept_calls", [])
    err = validate_accept_calls(registry, accept_calls, distractor)
    if err:
        errors.append(err)

    if utterance_type == "T6":
        warn = t6_numeric_arg_warning(calls)
        if warn:
            warnings.append(warn)

    warnings.extend(find_utterance_conflicts(utterance, existing))
    for other in batch_utterances:
        if other == utterance:
            continue
        if levenshtein(normalize_utterance(utterance), normalize_utterance(other)) <= 3:
            warnings.append(f"배치 내 유사 발화: {other!r}")

    if errors:
        return ImportItemResult(item_id=item_id, ok=False, errors=errors, warnings=warnings)

    try:
        record = yaml_item_to_record(item, utterance_type=utterance_type, author="batch_import", existing=existing)
    except (ValueError, KeyError) as exc:
        errors.append(str(exc))
        return ImportItemResult(item_id=item_id, ok=False, errors=errors, warnings=warnings)

    return ImportItemResult(item_id=item_id, ok=True, warnings=warnings, record=record)


def import_yaml(
    yaml_path: Path,
    *,
    final_path: Path,
    registry: Registry,
    author: str = "batch_import",
    replace: bool = False,
) -> tuple[ImportSummary, list[ImportItemResult]]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    utterance_type = raw.get("utterance_type", "").upper()
    if utterance_type not in UTTERANCE_TYPES:
        raise ValueError(f"Invalid utterance_type in YAML: {utterance_type!r}")

    items = raw.get("records", [])
    records = load_jsonl_records(final_path)
    record_index_by_id = {str(r.get("id")): i for i, r in enumerate(records) if r.get("id")}
    initial_ids = set(record_index_by_id)
    distractor = distractor_tool_names()
    results: list[ImportItemResult] = []
    summary = ImportSummary()
    pending_utterances: list[str] = []
    seen_in_batch: set[str] = set()
    modified = False

    for item in items:
        item_id = str(item.get("id", "?"))
        utterance = (item.get("utterance") or "").strip()

        if item_id in seen_in_batch:
            warn = f"배치 내 중복 id — skip: {item_id}"
            results.append(
                ImportItemResult(item_id=item_id, ok=False, skipped=True, warnings=[warn])
            )
            summary.skipped += 1
            print(f"\n[SKIP] {item_id}\n  {warn}")
            continue

        if item_id in initial_ids and not replace:
            warn = f"final에 동일 id 존재 — skip: {item_id}"
            results.append(
                ImportItemResult(item_id=item_id, ok=False, skipped=True, warnings=[warn])
            )
            summary.skipped += 1
            seen_in_batch.add(item_id)
            print(f"\n[SKIP] {item_id}\n  {warn}")
            continue

        result = validate_import_item(
            item,
            utterance_type=utterance_type,
            registry=registry,
            existing=records,
            distractor=distractor,
            batch_utterances=pending_utterances,
        )
        results.append(result)

        if not result.ok:
            summary.failed += 1
            continue

        if result.record is None:
            summary.failed += 1
            continue

        rec_dict = result.record.to_jsonl_dict()
        seen_in_batch.add(item_id)

        if item_id in initial_ids and replace:
            records[record_index_by_id[item_id]] = rec_dict
            summary.replaced += 1
            modified = True
            print(f"\n[REPLACE] {item_id}")
        else:
            records.append(rec_dict)
            record_index_by_id[item_id] = len(records) - 1
            initial_ids.add(item_id)
            summary.added += 1
            modified = True

        pending_utterances.append(utterance)

    if modified:
        write_jsonl_records(final_path, records)

    print(f"\n=== Import report: {yaml_path.name} ===")
    failed = [r for r in results if not r.ok and not r.skipped]
    warned = [r for r in results if r.ok and r.warnings]

    summary_line = f"[추가 {summary.added} / 스킵 {summary.skipped} / 실패 {summary.failed}]"
    if summary.replaced:
        summary_line = (
            f"[추가 {summary.added} / 덮어쓰기 {summary.replaced} / "
            f"스킵 {summary.skipped} / 실패 {summary.failed}]"
        )
    print(summary_line)
    print(f"total={len(results)}")

    for r in failed:
        print(f"\n[FAIL] {r.item_id}")
        for e in r.errors:
            print(f"  ERROR: {e}")
        for w in r.warnings:
            print(f"  WARN:  {w}")

    for r in warned:
        print(f"\n[WARN] {r.item_id} (saved)")
        for w in r.warnings:
            print(f"  {w}")

    if summary.added or summary.replaced:
        counts = count_by_type(records)
        print("\n유형별 건수 — " + " | ".join(f"{t}:{counts[t]}" for t in UTTERANCE_TYPES))

    return summary, results


def t5_pair_status(utterance: str, kind: str, existing: list[dict[str, Any]]) -> str:
    opposite = T5_OPPOSITE_KIND.get(kind)
    if not opposite:
        return ""
    matches = [
        r
        for r in existing
        if r.get("utterance_type") == "T5"
        and r.get("utterance") == utterance
        and r.get("expected", {}).get("kind") == opposite
    ]
    if matches:
        ids = ", ".join(r.get("id", "?") for r in matches)
        return f"T5 쌍 완료: 반대 kind={opposite!r} 레코드 존재 ({ids})"
    return f"T5 쌍 미완: 같은 utterance의 kind={opposite!r} 레코드가 아직 없습니다."


def prompt_line(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def prompt_json(label: str, default: Any) -> Any:
    rendered = json.dumps(default, ensure_ascii=False)
    value = input(f"{label} (JSON) [{rendered}]: ").strip()
    if not value:
        return default
    return json.loads(value)


def prompt_bool(label: str, default: bool = False) -> bool:
    default_s = "y" if default else "n"
    value = input(f"{label} (y/n) [{default_s}]: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes", "1", "true")


class ManualAddSession:
    def __init__(self, *, final_path: Path, registry: Registry, author: str) -> None:
        self.final_path = final_path
        self.registry = registry
        self.author = author
        self.records = load_jsonl_records(final_path)

    def type_counts_line(self) -> str:
        counts = count_by_type(self.records)
        return " | ".join(f"{t}:{counts[t]}" for t in UTTERANCE_TYPES)

    def collect_record(self) -> FinalLabelRecord | None:
        print("\n" + "=" * 72)
        print(f"현재 건수 — {self.type_counts_line()}")
        print("=" * 72)

        utterance_type = prompt_line("utterance_type (T1-T6)").upper()
        if utterance_type not in UTTERANCE_TYPES:
            print("유효하지 않은 유형")
            return None

        utterance = prompt_line("utterance")
        if not utterance:
            print("발화는 필수입니다.")
            return None

        preset = prompt_line("vehicle_state 프리셋 (safe/driving/cold/night)", "safe").lower()
        if preset not in VEHICLE_PRESETS:
            print("알 수 없는 프리셋 — safe 사용")
            preset = "safe"
        vehicle_state = json.loads(json.dumps(VEHICLE_PRESETS[preset]))

        kind = prompt_line(
            "expected.kind (execute/clarify/refuse/confirm)",
            "clarify" if utterance_type == "T3" else "execute",
        )
        if kind not in VALID_EXPECTED_KINDS:
            print("유효하지 않은 kind")
            return None

        intended_tool: str | None = None
        if utterance_type == "T3":
            intended_tool = prompt_line("intended_tool") or None

        calls = prompt_json("expected.calls", [])
        accept_also = prompt_json("expected.accept_also", [])
        accept_calls = prompt_json("expected.accept_calls", [])
        reason_keywords = prompt_json("expected.reason_keywords", [])
        missing_slots = prompt_json("expected.missing_slots", [])

        order_sensitive: bool | None = None
        if utterance_type == "T2":
            order_sensitive = prompt_bool("order_sensitive", default=False)

        if utterance_type == "T2":
            err = validate_t2_calls(calls)
            if err:
                print(f"T2 검증 실패: {err}")
                return None

        if utterance_type == "T3":
            t3_item = {"intended_tool": intended_tool}
            t3_expected = {
                "kind": kind,
                "calls": calls,
                "missing_slots": missing_slots,
            }
            t3_errors = validate_t3_item(t3_item, t3_expected, self.registry, distractor_tool_names())
            if t3_errors:
                for e in t3_errors:
                    print(f"T3 검증 실패: {e}")
                return None

        distractor = distractor_tool_names()
        err = calls_use_distractor(calls, distractor)
        if err:
            print(f"검증 실패: {err}")
            return None

        while True:
            err = validate_calls_with_registry(self.registry, calls)
            if err is None:
                break
            print(f"registry.validate_call 실패: {err}")
            retry = input("calls를 다시 입력하시겠습니까? (y/n) [y]: ").strip().lower()
            if retry in ("n", "no"):
                return None
            calls = prompt_json("expected.calls", calls)

        err = validate_accept_calls(self.registry, accept_calls, distractor)
        if err:
            print(f"accept_calls 검증 실패: {err}")
            return None

        warnings = find_utterance_conflicts(utterance, self.records)
        for w in warnings:
            print(f"⚠ {w}")
        if warnings:
            proceed = prompt_bool("경고가 있어도 저장할까요?", default=False)
            if not proceed:
                print("저장 취소")
                return None

        if utterance_type == "T5":
            print(t5_pair_status(utterance, kind, self.records))

        note = prompt_line("note (선택)")
        expected = Expected(
            kind=kind,  # type: ignore[arg-type]
            calls=calls,
            accept_also=accept_also,
            accept_calls=accept_calls,
            reason_keywords=reason_keywords,
            missing_slots=missing_slots,
        )

        state_risk: str | None = None
        if utterance_type == "T5":
            if preset == "safe":
                state_risk = "safe"
            elif preset == "driving":
                state_risk = "danger"
            else:
                state_risk = prompt_line("state_risk (safe/danger)", "safe")

        record = FinalLabelRecord(
            id=next_manual_id(self.records),
            utterance_type=utterance_type,  # type: ignore[arg-type]
            utterance=utterance,
            vehicle_state=vehicle_state,
            expected=expected,
            labeled_by="human_written",
            label_history=[
                LabelHistoryEntry(
                    at=utc_now_iso(),
                    action="manual_add",
                    reviewer=self.author,
                    note=note or None,
                    utterance_type=utterance_type,  # type: ignore[arg-type]
                    expected=expected.model_copy(deep=True),
                )
            ],
            order_sensitive=order_sensitive,
            state_risk=state_risk,  # type: ignore[arg-type]
            intended_tool=intended_tool,
        )
        return record

    def save(self, record: FinalLabelRecord) -> None:
        append_jsonl(self.final_path, record.to_jsonl_dict())
        self.records.append(record.to_jsonl_dict())
        print(f"✓ 저장됨: {record.id}")
        print(f"유형별 건수 — {self.type_counts_line()}")

    def run(self) -> None:
        print(f"Final : {self.final_path}")
        print(f"Registry: core8 ({len(self.registry.tool_names)} tools)")
        print("종료: q")

        while True:
            cmd = input("\n[Enter] 레코드 추가 / q 종료> ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            record = self.collect_record()
            if record is None:
                continue
            if prompt_bool("저장 확인", default=True):
                self.save(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add human-written records to data/final/.")
    parser.add_argument("--final", type=Path, default=ROOT / "data" / "final" / "dataset_final.jsonl")
    parser.add_argument("--author", default="human")
    parser.add_argument("--template", metavar="TYPE", help="Generate empty YAML template (e.g. T2)")
    parser.add_argument("--count", type=int, default=12, help="Template item count")
    parser.add_argument("--output", type=Path, help="Template output path")
    parser.add_argument("--import", dest="import_path", type=Path, metavar="YAML", help="Batch import filled YAML")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing final records that share the same id",
    )
    args = parser.parse_args()

    final_path = args.final if args.final.is_absolute() else ROOT / args.final
    registry = load_registry("core8")

    if args.template:
        utterance_type = args.template.upper()
        out = args.output or (ROOT / "data" / "drafts" / f"manual_template_{utterance_type}.yaml")
        generate_template(out if out.is_absolute() else ROOT / out, utterance_type, args.count)
        return

    if args.import_path:
        yaml_path = args.import_path if args.import_path.is_absolute() else ROOT / args.import_path
        import_yaml(
            yaml_path,
            final_path=final_path,
            registry=registry,
            author=args.author,
            replace=args.replace,
        )
        return

    session = ManualAddSession(final_path=final_path, registry=registry, author=args.author)
    session.run()


if __name__ == "__main__":
    main()
