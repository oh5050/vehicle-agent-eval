"""Tests for scripts/manual_add.py helpers."""

from __future__ import annotations

from scripts.manual_add import (
    find_utterance_conflicts,
    levenshtein,
    normalize_utterance,
    t5_pair_status,
    validate_t2_calls,
)
from src.registry import load_registry
from scripts.manual_add import validate_calls_with_registry


def test_levenshtein():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1


def test_normalize_utterance_strips_punctuation():
    assert normalize_utterance("운전석 창 열어줘!") == normalize_utterance("운전석창열어줘")


def test_find_utterance_conflicts_exact():
    existing = [{"id": "x1", "utterance": "트렁크 열어줘"}]
    warnings = find_utterance_conflicts("트렁크 열어줘", existing)
    assert any("완전 일치" in w for w in warnings)


def test_find_utterance_conflicts_similar():
    existing = [{"id": "x1", "utterance": "트렁크열어줘"}]
    warnings = find_utterance_conflicts("트렁크 열어", existing)
    assert warnings


def test_validate_t2_calls():
    assert validate_t2_calls([{"tool": "a", "args": {}}]) is not None
    assert validate_t2_calls([{"tool": "a", "args": {}}, {"tool": "a", "args": {}}]) is not None
    assert validate_t2_calls(
        [{"tool": "climate_set", "args": {"temperature": 22}}, {"tool": "media_control", "args": {"action": "play"}}]
    ) is None


def test_validate_calls_with_registry():
    registry = load_registry("core8")
    err = validate_calls_with_registry(
        registry, [{"tool": "trunk_control", "args": {"action": "open"}}]
    )
    assert err is None
    err_bad = validate_calls_with_registry(registry, [{"tool": "trunk_control", "args": {}}])
    assert err_bad is not None


def test_t5_pair_status_missing():
    msg = t5_pair_status("트렁크 열어줘", "execute", [])
    assert "미완" in msg


def test_t5_pair_status_complete():
    existing = [
        {
            "id": "m1",
            "utterance_type": "T5",
            "utterance": "트렁크 열어줘",
            "expected": {"kind": "confirm"},
        }
    ]
    msg = t5_pair_status("트렁크 열어줘", "execute", existing)
    assert "완료" in msg
