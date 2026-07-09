"""Tests for manual_add batch mode."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.manual_add import (
    build_template_header,
    distractor_tool_names,
    generate_template,
    import_yaml,
    load_jsonl_records,
    t6_numeric_arg_warning,
    validate_import_item,
    validate_t3_item,
)
from src.registry import load_registry


def _t3_item(**overrides: object) -> dict:
    base = {
        "id": "t3_001",
        "utterance": "창문 열어줘",
        "intended_tool": "window_control",
        "vehicle_state_preset": "safe",
        "expected": {
            "kind": "clarify",
            "calls": [],
            "missing_slots": ["window"],
            "accept_also": [],
            "accept_calls": [],
        },
    }
    expected_overrides = overrides.pop("expected", None)
    base.update(overrides)
    if expected_overrides:
        base["expected"] = {**base["expected"], **expected_overrides}  # type: ignore[dict-item]
    return base


def _run_t3_import(tmp_path: Path, records: list[dict], *, final_path: Path | None = None):
    yaml_path = tmp_path / "t3_import.yaml"
    out = final_path or tmp_path / "final.jsonl"
    payload = {"utterance_type": "T3", "records": records}
    yaml_path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")
    return import_yaml(yaml_path, final_path=out, registry=load_registry("core8"))


def test_import_t3_violation_clarify_calls_not_empty(tmp_path: Path):
    """[B-1] kind=clarify이면 calls는 빈 배열."""
    bad = _t3_item(
        id="t3_bad_calls",
        expected={"kind": "clarify", "calls": [{"tool": "window_control", "args": {}}], "missing_slots": ["window"]},
    )
    summary, results = _run_t3_import(tmp_path, [bad])
    assert summary.failed == 1
    assert summary.added == 0
    assert results[0].item_id == "t3_bad_calls"
    assert not results[0].ok
    assert any("calls가 비어" in e for e in results[0].errors)
    assert load_jsonl_records(tmp_path / "final.jsonl") == []


def test_import_t3_violation_intended_tool_not_in_registry(tmp_path: Path):
    """[B-2] intended_tool은 registry에 존재해야 함."""
    bad = _t3_item(id="t3_bad_tool", intended_tool="nonexistent_tool")
    summary, results = _run_t3_import(tmp_path, [bad])
    assert summary.failed == 1
    assert results[0].item_id == "t3_bad_tool"
    assert any("registry에 없습니다" in e for e in results[0].errors)
    assert load_jsonl_records(tmp_path / "final.jsonl") == []


def test_import_t3_violation_missing_slot_not_in_schema(tmp_path: Path):
    """[B-3] missing_slots 항목은 intended_tool 스키마의 required/required_one_of에 속해야 함."""
    bad = _t3_item(id="t3_bad_slot", expected={"kind": "clarify", "calls": [], "missing_slots": ["zone"]})
    summary, results = _run_t3_import(tmp_path, [bad])
    assert summary.failed == 1
    assert results[0].item_id == "t3_bad_slot"
    assert any("required/required_one_of" in e for e in results[0].errors)
    assert load_jsonl_records(tmp_path / "final.jsonl") == []


def test_import_t3_violation_partial_required_one_of(tmp_path: Path):
    """[B-4] required_one_of 툴은 missing_slots가 그룹 전체를 포함해야 함."""
    bad = _t3_item(
        id="t3_bad_one_of",
        intended_tool="climate_set",
        utterance="좀 더워",
        expected={"kind": "clarify", "calls": [], "missing_slots": ["temperature"]},
    )
    summary, results = _run_t3_import(tmp_path, [bad])
    assert summary.failed == 1
    assert results[0].item_id == "t3_bad_one_of"
    assert any("required_one_of" in e for e in results[0].errors)
    assert load_jsonl_records(tmp_path / "final.jsonl") == []


def test_import_t3_violation_invalid_accept_calls(tmp_path: Path):
    """[B-5] accept_calls 중첩 리스트의 각 call은 registry.validate_call 통과."""
    bad = _t3_item(
        id="t3_bad_accept_calls",
        expected={
            "kind": "clarify",
            "calls": [],
            "missing_slots": ["window"],
            "accept_calls": [[{"tool": "window_control", "args": {"window": "driver", "action": "invalid"}}]],
        },
    )
    summary, results = _run_t3_import(tmp_path, [bad])
    assert summary.failed == 1
    assert results[0].item_id == "t3_bad_accept_calls"
    assert any("accept_calls" in e for e in results[0].errors)
    assert load_jsonl_records(tmp_path / "final.jsonl") == []


def test_import_t3_mixed_batch_saves_only_valid(tmp_path: Path):
    """[B-6] 위반 건은 제외하고 유효 건만 저장."""
    good = _t3_item(id="t3_good", utterance="창문 열어줘")
    bad = _t3_item(id="t3_bad", expected={"kind": "clarify", "calls": [{"tool": "x", "args": {}}], "missing_slots": ["window"]})
    summary, results = _run_t3_import(tmp_path, [good, bad])
    assert summary.added == 1
    assert summary.failed == 1
    rows = load_jsonl_records(tmp_path / "final.jsonl")
    assert len(rows) == 1
    assert rows[0]["id"] == "t3_good"


def test_build_template_header_t2():
    header = build_template_header("T2")
    assert "서로 다른 2개 이상 툴" in header
    assert "order_sensitive" in header


def test_build_template_header_t3():
    header = build_template_header("T3")
    assert "intended_tool" in header
    assert "required_one_of" in header


def test_generate_template_t3_includes_intended_tool(tmp_path: Path):
    path = tmp_path / "manual_template_T3.yaml"
    generate_template(path, "T3", 2)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["records"][0]["intended_tool"] == ""
    assert data["records"][0]["expected"]["missing_slots"] == []


def test_validate_t3_window_control_ok():
    registry = load_registry("core8")
    errors = validate_t3_item(
        _t3_item(),
        _t3_item()["expected"],
        registry,
        distractor_tool_names(),
    )
    assert errors == []


def test_validate_t3_clarify_with_calls_fails():
    registry = load_registry("core8")
    item = _t3_item(expected={"kind": "clarify", "calls": [{"tool": "window_control", "args": {}}], "missing_slots": ["window"]})
    errors = validate_t3_item(item, item["expected"], registry, distractor_tool_names())
    assert any("calls가 비어" in e for e in errors)


def test_validate_t3_missing_intended_tool():
    registry = load_registry("core8")
    item = _t3_item(intended_tool="")
    errors = validate_t3_item(item, item["expected"], registry, distractor_tool_names())
    assert any("intended_tool" in e for e in errors)


def test_validate_t3_invalid_missing_slot():
    registry = load_registry("core8")
    item = _t3_item(expected={"kind": "clarify", "calls": [], "missing_slots": ["zone"]})
    errors = validate_t3_item(item, item["expected"], registry, distractor_tool_names())
    assert any("required/required_one_of" in e for e in errors)


def test_validate_t3_climate_partial_one_of_fails():
    registry = load_registry("core8")
    item = _t3_item(
        intended_tool="climate_set",
        utterance="좀 더워",
        expected={"kind": "clarify", "calls": [], "missing_slots": ["temperature"]},
    )
    errors = validate_t3_item(item, item["expected"], registry, distractor_tool_names())
    assert any("required_one_of" in e for e in errors)
    assert any("일부만 누락" in e for e in errors)


def test_validate_t3_climate_full_one_of_ok():
    registry = load_registry("core8")
    item = _t3_item(
        intended_tool="climate_set",
        utterance="좀 더워",
        expected={
            "kind": "clarify",
            "calls": [],
            "missing_slots": ["temperature", "fan_level", "mode"],
        },
    )
    errors = validate_t3_item(item, item["expected"], registry, distractor_tool_names())
    assert errors == []


def test_validate_import_item_t3_integration():
    registry = load_registry("core8")
    item = _t3_item(
        intended_tool="climate_set",
        expected={"kind": "clarify", "calls": [], "missing_slots": ["temperature"]},
    )
    result = validate_import_item(
        item,
        utterance_type="T3",
        registry=registry,
        existing=[],
        distractor=distractor_tool_names(),
        batch_utterances=[],
    )
    assert not result.ok
    assert any("required_one_of" in e for e in result.errors)


def test_generate_template(tmp_path: Path):
    path = tmp_path / "manual_template_T2.yaml"
    generate_template(path, "T2", 3)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["utterance_type"] == "T2"
    assert len(data["records"]) == 3
    assert data["records"][0]["utterance"] == ""
    assert data["records"][0]["expected"]["accept_calls"] == []


def test_validate_import_item_t4_empty_calls(tmp_path: Path):
    registry = load_registry("core8")
    item = {
        "id": "x1",
        "utterance": "마사지 켜줘",
        "expected": {"kind": "refuse", "calls": [{"tool": "climate_set", "args": {}}]},
        "vehicle_state_preset": "safe",
    }
    result = validate_import_item(
        item,
        utterance_type="T4",
        registry=registry,
        existing=[],
        distractor=distractor_tool_names(),
        batch_utterances=[],
    )
    assert not result.ok
    assert any("calls가 비어" in e for e in result.errors)


def test_validate_import_item_distractor_rejected():
    registry = load_registry("core8")
    distractor = distractor_tool_names()
    assert distractor
    tool = next(iter(distractor))
    item = {
        "id": "x2",
        "utterance": "테스트",
        "expected": {"kind": "execute", "calls": [{"tool": tool, "args": {}}]},
        "vehicle_state_preset": "safe",
    }
    result = validate_import_item(
        item,
        utterance_type="T1",
        registry=registry,
        existing=[],
        distractor=distractor,
        batch_utterances=[],
    )
    assert not result.ok
    assert any("distractor" in e for e in result.errors)


def test_t6_numeric_arg_warning():
    assert t6_numeric_arg_warning([{"tool": "climate_set", "args": {"temperature": 22}}])


def test_validate_import_item_accept_calls_valid():
    registry = load_registry("core8")
    item = {
        "id": "x3",
        "utterance": "운전석 창 열어줘",
        "expected": {
            "kind": "execute",
            "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
            "accept_calls": [
                [{"tool": "window_control", "args": {"window": "all", "action": "open"}}],
            ],
        },
        "vehicle_state_preset": "safe",
    }
    result = validate_import_item(
        item,
        utterance_type="T1",
        registry=registry,
        existing=[],
        distractor=distractor_tool_names(),
        batch_utterances=[],
    )
    assert result.ok


def test_validate_import_item_accept_calls_invalid():
    registry = load_registry("core8")
    item = {
        "id": "x4",
        "utterance": "운전석 창 열어줘",
        "expected": {
            "kind": "execute",
            "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
            "accept_calls": [
                [{"tool": "window_control", "args": {"window": "driver", "action": "invalid"}}],
            ],
        },
        "vehicle_state_preset": "safe",
    }
    result = validate_import_item(
        item,
        utterance_type="T1",
        registry=registry,
        existing=[],
        distractor=distractor_tool_names(),
        batch_utterances=[],
    )
    assert not result.ok
    assert any("accept_calls" in e for e in result.errors)


def test_import_yaml_batch(tmp_path: Path):
    registry = load_registry("core8")
    final_path = tmp_path / "final.jsonl"
    yaml_path = tmp_path / "manual_template_T1.yaml"
    payload = {
        "utterance_type": "T1",
        "records": [
            {
                "id": "manual_T1_001",
                "utterance": "운전석 창 열어줘",
                "vehicle_state_preset": "safe",
                "note": "",
                "expected": {
                    "kind": "execute",
                    "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
                    "accept_also": [],
                    "order_sensitive": None,
                },
            },
            {
                "id": "manual_T1_002",
                "utterance": "",
                "vehicle_state_preset": "safe",
                "note": "",
                "expected": {"kind": "execute", "calls": [], "accept_also": [], "order_sensitive": None},
            },
        ],
    }
    yaml_path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")

    saved, results = import_yaml(yaml_path, final_path=final_path, registry=registry)
    assert saved.added == 1
    assert saved.skipped == 0
    assert saved.failed == 1
    assert len(results) == 2
    assert not results[1].ok
    rows = load_jsonl_records(final_path)
    assert len(rows) == 1


def _sample_import_yaml(tmp_path: Path) -> tuple[Path, Path]:
    registry = load_registry("core8")
    final_path = tmp_path / "final.jsonl"
    yaml_path = tmp_path / "manual_template_T1.yaml"
    payload = {
        "utterance_type": "T1",
        "records": [
            {
                "id": "manual_T1_001",
                "utterance": "운전석 창 열어줘",
                "vehicle_state_preset": "safe",
                "note": "",
                "expected": {
                    "kind": "execute",
                    "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
                    "accept_also": [],
                    "accept_calls": [],
                    "order_sensitive": None,
                },
            },
        ],
    }
    yaml_path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")
    return yaml_path, final_path


def test_import_yaml_idempotent(tmp_path: Path):
    yaml_path, final_path = _sample_import_yaml(tmp_path)
    registry = load_registry("core8")

    summary1, _ = import_yaml(yaml_path, final_path=final_path, registry=registry)
    assert summary1.added == 1
    assert len(load_jsonl_records(final_path)) == 1

    summary2, results2 = import_yaml(yaml_path, final_path=final_path, registry=registry)
    assert summary2.added == 0
    assert summary2.skipped == 1
    assert summary2.failed == 0
    assert results2[0].skipped
    assert len(load_jsonl_records(final_path)) == 1


def test_import_yaml_replace(tmp_path: Path):
    yaml_path, final_path = _sample_import_yaml(tmp_path)
    registry = load_registry("core8")

    import_yaml(yaml_path, final_path=final_path, registry=registry)

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    payload["records"][0]["utterance"] = "운전석 창 닫아줘"
    payload["records"][0]["expected"]["calls"] = [
        {"tool": "window_control", "args": {"window": "driver", "action": "close"}}
    ]
    yaml_path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")

    summary, _ = import_yaml(yaml_path, final_path=final_path, registry=registry, replace=True)
    assert summary.replaced == 1
    assert summary.added == 0
    assert summary.skipped == 0

    rows = load_jsonl_records(final_path)
    assert len(rows) == 1
    assert rows[0]["utterance"] == "운전석 창 닫아줘"
