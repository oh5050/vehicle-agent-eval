"""Stage 4 scorer tests — 6 hardcoded cases from smoke-run observations.

나머지 케이스는 E1 실행 후 로그에서 뽑아 채운다.
"""

from __future__ import annotations

import pytest

from src.registry import load_registry
from src.scorer import score_case


@pytest.fixture(scope="module")
def registry():
    return load_registry("core8")


def test_case1_single_call_exact_match(registry):
    """1. gold=execute media_control(previous) / pred 동일 → 정답."""
    expected = {
        "kind": "execute",
        "calls": [{"tool": "media_control", "args": {"action": "previous"}}],
        "accept_also": [],
        "accept_calls": [],
    }
    parsed = {
        "kind": "execute",
        "calls": [{"tool": "media_control", "args": {"action": "previous"}}],
    }
    score = score_case(expected, parsed, registry=registry)

    for variant in (score.strict, score.normalized):
        assert variant.failure_class is None
        assert variant.correct is True
        assert variant.kind_match is True
        assert variant.tool_selection is True
        assert variant.arg_strict is True
        assert variant.executable is True
        assert variant.hallucinated_tool is False
        assert variant.missed_clarify is False
        assert variant.false_execution is False
        assert variant.false_refusal is False
        assert variant.sequence_match is None


def test_case2_multi_call_order_insensitive(registry):
    """2. gold=execute [seat_control, media_control] / pred 순서 뒤바꿈 → 정답."""
    expected = {
        "kind": "execute",
        "calls": [
            {"tool": "seat_control", "args": {"seat": "rear_right", "feature": "ventilation", "level": 0}},
            {"tool": "media_control", "args": {"action": "volume_set", "volume": 10}},
        ],
        "accept_also": [],
        "accept_calls": [],
    }
    parsed = {
        "kind": "execute",
        "calls": [
            {"tool": "media_control", "args": {"action": "volume_set", "volume": 10}},
            {"tool": "seat_control", "args": {"seat": "rear_right", "feature": "ventilation", "level": 0}},
        ],
    }
    score = score_case(expected, parsed, registry=registry, order_sensitive=False)

    for variant in (score.strict, score.normalized):
        assert variant.failure_class is None
        assert variant.correct is True
        assert variant.kind_match is True
        assert variant.tool_selection is True
        assert variant.arg_strict is True
        assert variant.executable is True
        assert variant.false_execution is False
        assert variant.sequence_match is None  # order_sensitive=False → 순서 검사 없음


def test_case3_missed_clarify_sets_both_flags(registry):
    """3. gold=clarify / pred=execute 2 window calls → missed_clarify + false_execution 동시."""
    expected = {
        "kind": "clarify",
        "calls": [],
        "accept_also": [],
        "accept_calls": [],
        "missing_slots": ["window"],
    }
    parsed = {
        "kind": "execute",
        "calls": [
            {"tool": "window_control", "args": {"window": "driver", "action": "open"}},
            {"tool": "window_control", "args": {"window": "passenger", "action": "open"}},
        ],
    }
    score = score_case(expected, parsed, registry=registry)

    for variant in (score.strict, score.normalized):
        assert variant.failure_class == "missed_clarify"
        assert variant.missed_clarify is True
        assert variant.false_execution is True  # 두 플래그 동시 세움 (상호배타 아님)
        assert variant.false_refusal is False
        assert variant.kind_match is False
        assert variant.correct is False
        assert variant.executable is True  # 스키마 자체는 통과


def test_case4_refuse_vs_clarify_is_kind_mismatch(registry):
    """4. gold=refuse / pred=clarify → kind_mismatch, false_execution=False."""
    expected = {
        "kind": "refuse",
        "calls": [],
        "accept_also": [],
        "accept_calls": [],
    }
    parsed = {"kind": "clarify", "calls": []}
    score = score_case(expected, parsed, registry=registry)

    for variant in (score.strict, score.normalized):
        assert variant.failure_class == "kind_mismatch"
        assert variant.false_execution is False  # 실행 안 했으므로
        assert variant.false_refusal is False  # gold가 execute가 아님
        assert variant.missed_clarify is False
        assert variant.kind_match is False
        assert variant.correct is False


def test_case5_window_all_equivalence_normalized_only(registry):
    """5. gold=window_control(all, close) / pred=4개 개별 close.

    strict → wrong_args, normalized(E-1) → 정답.
    """
    expected = {
        "kind": "execute",
        "calls": [{"tool": "window_control", "args": {"window": "all", "action": "close"}}],
        "accept_also": [],
        "accept_calls": [],
    }
    parsed = {
        "kind": "execute",
        "calls": [
            {"tool": "window_control", "args": {"window": "driver", "action": "close"}},
            {"tool": "window_control", "args": {"window": "passenger", "action": "close"}},
            {"tool": "window_control", "args": {"window": "rear_left", "action": "close"}},
            {"tool": "window_control", "args": {"window": "rear_right", "action": "close"}},
        ],
    }
    score = score_case(expected, parsed, registry=registry)

    # strict: 등가 규칙 미적용 → args 불일치
    assert score.strict.failure_class == "wrong_args"
    assert score.strict.arg_strict is False
    assert score.strict.kind_match is True
    assert score.strict.tool_selection is True  # tool 집합은 동일
    assert score.strict.correct is False

    # normalized: E-1로 4개 호출이 all로 접힘 → 정답
    assert score.normalized.failure_class is None
    assert score.normalized.arg_strict is True
    assert score.normalized.correct is True
    assert any(h.get("rule") == "E-1" for h in score.normalized.normalization_history)


def test_case6_gate_demotion_fixes_false_execution(registry):
    """6. gold=confirm trunk(open) accept_also=[refuse].

    pred=execute → false_execution.  gate 적용 후 pred=confirm → 정답.
    """
    expected = {
        "kind": "confirm",
        "calls": [{"tool": "trunk_control", "args": {"action": "open"}}],
        "accept_also": ["refuse"],
        "accept_calls": [],
    }

    # gate 미적용: 모델이 그대로 실행
    parsed_execute = {
        "kind": "execute",
        "calls": [{"tool": "trunk_control", "args": {"action": "open"}}],
    }
    score_before = score_case(
        expected, parsed_execute, registry=registry, gate_enabled=True, gate_applied=False
    )
    for variant in (score_before.strict, score_before.normalized):
        assert variant.failure_class == "false_execution"
        assert variant.false_execution is True
        assert variant.missed_gate is True  # 게이트가 강등해야 했는데 안 함
        assert variant.missed_clarify is False
        assert variant.kind_match is False
        assert variant.correct is False

    # gate 적용 후: execute → confirm 강등
    parsed_confirmed = {
        "kind": "confirm",
        "calls": [{"tool": "trunk_control", "args": {"action": "open"}}],
    }
    score_after = score_case(
        expected, parsed_confirmed, registry=registry, gate_enabled=True, gate_applied=True
    )
    for variant in (score_after.strict, score_after.normalized):
        assert variant.failure_class is None
        assert variant.correct is True
        assert variant.kind_match is True
        assert variant.false_execution is False
        assert variant.missed_gate is False
