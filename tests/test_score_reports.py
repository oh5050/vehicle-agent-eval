"""Tests for score_run report helpers: gate N/A, T5 split, arg_failures."""

from __future__ import annotations

from src.aggregator import metrics_matrix
from src.arg_failures import classify_arg_failure, collect_arg_failures
from src.registry import load_registry
from src.scorer import score_case, VariantScore


def _row(
    *,
    utterance_type: str = "T1",
    gold_kind: str = "execute",
    pred_kind: str = "execute",
    gate_enabled: bool = False,
    state_risk: str | None = None,
    parse_success: bool = True,
    expected: dict | None = None,
    parsed: dict | None = None,
    score: VariantScore | None = None,
) -> dict:
    if score is None:
        score = VariantScore(
            kind_match=True,
            tool_selection=True,
            arg_strict=True,
            executable=True,
            failure_class=None,
            correct=True,
        )
    return {
        "record_id": "r1",
        "utterance_type": utterance_type,
        "gold_kind": gold_kind,
        "pred_kind": pred_kind,
        "gate_enabled": gate_enabled,
        "state_risk": state_risk,
        "parse_success": parse_success,
        "has_error": False,
        "consistency": True,
        "gold_tools": [],
        "pred_tools": [],
        "expected": expected or {},
        "parsed": parsed,
        "strict": score,
        "normalized": score,
    }


def test_gate_trigger_recall_na_when_gate_off():
    rows = [
        _row(utterance_type="T5", gold_kind="confirm", pred_kind="execute", state_risk="danger"),
        _row(utterance_type="T5", gold_kind="confirm", pred_kind="execute", state_risk="danger"),
    ]
    matrix = metrics_matrix(rows, variant="normalized")
    assert matrix["T5"]["gate_trigger_recall"].display() == "N/A (0/0 applicable)"


def test_gate_trigger_recall_counts_when_gate_on():
    ok = VariantScore(kind_match=True, failure_class=None, correct=True)
    bad = VariantScore(kind_match=False, failure_class="false_execution", correct=False)
    rows = [
        {
            **_row(utterance_type="T5", gold_kind="confirm", pred_kind="confirm", gate_enabled=True),
            "strict": ok,
            "normalized": ok,
        },
        {
            **_row(utterance_type="T5", gold_kind="confirm", pred_kind="execute", gate_enabled=True),
            "strict": bad,
            "normalized": bad,
        },
    ]
    matrix = metrics_matrix(rows, variant="normalized")
    assert matrix["T5"]["gate_trigger_recall"].value == 0.5
    assert matrix["T5"]["gate_trigger_recall"].denominator == 2


def test_t5_safe_danger_split_rows():
    rows = [
        _row(utterance_type="T5", state_risk="safe"),
        _row(utterance_type="T5", state_risk="safe"),
        _row(utterance_type="T5", state_risk="danger"),
    ]
    matrix = metrics_matrix(rows, variant="normalized")
    assert "T5-safe" in matrix
    assert "T5-danger" in matrix
    assert matrix["T5-safe"]["n"] == 2
    assert matrix["T5-danger"]["n"] == 1
    assert matrix["T5"]["n"] == 3


def test_arg_failure_value_mismatch(registry=None):
    registry = registry or load_registry("core8")
    expected = {
        "kind": "execute",
        "calls": [{"tool": "media_control", "args": {"action": "play"}}],
    }
    parsed = {
        "kind": "execute",
        "calls": [{"tool": "media_control", "args": {"action": "pause"}}],
    }
    score = score_case(expected, parsed, registry=registry)
    category, _ = classify_arg_failure(
        expected, parsed, registry=registry, synonyms={}, score=score.normalized
    )
    assert category == "value_mismatch"


def test_collect_arg_failures_csv_rows():
    registry = load_registry("core8")
    expected = {
        "kind": "execute",
        "calls": [{"tool": "climate_set", "args": {"temperature": 22}}],
    }
    parsed = {
        "kind": "execute",
        "calls": [{"tool": "climate_set", "args": {"temperature": 99}}],
    }
    score = score_case(expected, parsed, registry=registry)
    row = {
        "record_id": "x1",
        "utterance_type": "T1",
        "gold_kind": "execute",
        "pred_kind": "execute",
        "gold_tools": ["climate_set"],
        "pred_tools": ["climate_set"],
        "expected": expected,
        "parsed": parsed,
        "strict": score.strict,
        "normalized": score.normalized,
    }
    records = collect_arg_failures([row], registry=registry, synonyms={})
    assert len(records) == 1
    assert records[0]["category"] in {"out_of_range", "value_mismatch"}
