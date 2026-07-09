"""Tests for scripts/validate_dataset.py."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.validate_dataset import (
    Violation,
    find_duplicate_utterances,
    label_source,
    validate_dataset,
    validate_record,
    write_csv,
)
from src.registry import load_registry


def _record(**kwargs: object) -> dict:
    base = {
        "id": "r1",
        "utterance_type": "T1",
        "utterance": "운전석 창 열어줘",
        "labeled_by": "human",
        "expected": {
            "kind": "execute",
            "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
            "accept_also": [],
            "accept_calls": [],
            "missing_slots": [],
        },
    }
    expected_overrides = kwargs.pop("expected", None)
    base.update(kwargs)
    if expected_overrides:
        base["expected"] = {**base["expected"], **expected_overrides}  # type: ignore[dict-item]
    return base


def test_label_source_llm_draft():
    assert label_source({"source_labeled_by": "llm_draft", "labeled_by": "human"}) == "llm_draft"


def test_label_source_human():
    assert label_source({"labeled_by": "human_written"}) == "human_written"


def test_c1_execute_empty_calls():
    registry = load_registry("core8")
    rec = _record(expected={"kind": "execute", "calls": []})
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("C1:" in v.violation_reason for v in violations)


def test_c2_clarify_with_calls():
    registry = load_registry("core8")
    rec = _record(
        utterance_type="T3",
        expected={"kind": "clarify", "calls": [{"tool": "window_control", "args": {}}], "missing_slots": []},
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("C2:" in v.violation_reason for v in violations)


def test_c3_invalid_call():
    registry = load_registry("core8")
    rec = _record(
        expected={
            "kind": "execute",
            "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "invalid"}}],
        }
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any(v.violation_reason.startswith("C3:") for v in violations)


def test_c4_duplicate_utterances():
    a = _record(id="a1", utterance="창문 열어줘!")
    b = _record(id="a2", utterance="창문 열어줘")
    violations = find_duplicate_utterances([a, b])
    assert len(violations) == 2
    assert all("C4:" in v.violation_reason for v in violations)


def test_c4_skips_t5_same_utterance():
    a = _record(
        id="t5_safe",
        utterance_type="T5",
        utterance="트렁크 열어줘",
        expected={"kind": "execute", "calls": [{"tool": "trunk_control", "args": {"action": "open"}}]},
    )
    b = _record(
        id="t5_danger",
        utterance_type="T5",
        utterance="트렁크 열어줘",
        expected={"kind": "confirm", "calls": [{"tool": "trunk_control", "args": {"action": "open"}}]},
    )
    violations = find_duplicate_utterances([a, b])
    assert violations == []


def test_t1_numeric_arg_without_digit_in_utterance():
    registry = load_registry("core8")
    rec = _record(
        utterance="운전석 온도 좀 맞춰줘",
        expected={
            "kind": "execute",
            "calls": [{"tool": "climate_set", "args": {"temperature": 22, "mode": "cool"}}],
        },
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("T1/T2:" in v.violation_reason and "숫자" in v.violation_reason for v in violations)


def test_t6_command_pattern():
    registry = load_registry("core8")
    rec = _record(
        utterance_type="T6",
        utterance="에어컨 켜줘",
        expected={"kind": "execute", "calls": [], "missing_slots": []},
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("T6:" in v.violation_reason and "켜줘" in v.violation_reason for v in violations)


def test_t5_state_exposure_pattern():
    registry = load_registry("core8")
    rec = _record(
        utterance_type="T5",
        utterance="주행 중인데 트렁크 열어줘",
        expected={
            "kind": "execute",
            "calls": [{"tool": "trunk_control", "args": {"action": "open"}}],
        },
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("T5:" in v.violation_reason for v in violations)


def test_t5_clean_utterance_passes():
    registry = load_registry("core8")
    rec = _record(
        utterance_type="T5",
        utterance="트렁크 열어줘",
        expected={
            "kind": "execute",
            "calls": [{"tool": "trunk_control", "args": {"action": "open"}}],
        },
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert not any("T5:" in v.violation_reason for v in violations)


def test_t4_calls_not_empty():
    registry = load_registry("core8")
    rec = _record(
        utterance_type="T4",
        utterance="마사지 켜줘",
        expected={"kind": "refuse", "calls": [{"tool": "climate_set", "args": {}}]},
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any("T4:" in v.violation_reason for v in violations)


def test_t3_reuses_validator():
    registry = load_registry("core8")
    rec = _record(
        id="t3x",
        utterance_type="T3",
        utterance="창문 열어줘",
        expected={"kind": "clarify", "calls": [], "missing_slots": ["zone"]},
    )
    violations = validate_record(rec, registry=registry, distractor=set())
    assert any(v.violation_reason.startswith("T3:") for v in violations)


def test_write_csv(tmp_path: Path):
    path = tmp_path / "violations.csv"
    rows = [
        Violation("id1", "T1", "human_written", "C1: test"),
    ]
    write_csv(path, rows)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert data[0]["id"] == "id1"
    assert data[0]["violation_reason"] == "C1: test"


def test_validate_dataset_no_auto_delete(tmp_path: Path):
    """Validator is read-only — input file unchanged."""
    path = tmp_path / "final.jsonl"
    rec = _record(id="keep_me")
    import json

    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    validate_dataset([rec])
    assert path.read_text(encoding="utf-8") == before
