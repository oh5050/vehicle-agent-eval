"""Tests for draft matrix sampling and post-filters (no Ollama)."""

from __future__ import annotations

import copy

import pytest

from scripts.gen_drafts import generate_drafts
from src.draft_matrix import (
    apply_post_filters,
    check_unrecoverable_args,
    collect_tool_names_from_records,
    dedupe_records,
    derive_t3_from_t1,
    enumerate_tool_arg_combos,
    make_t5_pairs_from_utterances,
    omit_eligible_slots,
    parse_utterance_json,
    recoverable_numeric_values,
    sample_compound_slots,
    sample_slots,
    validate_utterance,
)
from src.draft_schema import VALID_EXPECTED_KINDS, DraftRecord, DraftStyle, Expected
from src.registry import load_registry


@pytest.fixture()
def config():
    return {
        "seed": 42,
        "registry": {"variant": "core8"},
        "vehicle_state_default": {
            "speed_kmh": 0,
            "gear": "P",
            "battery_pct": 72,
            "outside_temp_c": 3,
            "passengers": {"front": True, "rear": False},
        },
        "types": {
            "T1": {"count": 6, "max_risk_tier": 1},
            "T2": {"count": 4, "max_risk_tier": 1},
            "T3": {"count": 2, "source_type": "T1"},
            "T4": {"count": 2},
            "T5": {"count": 4, "min_state_pairs": 2},
            "T6": {"count": 2},
        },
        "styles": {
            "directness": ["direct", "indirect"],
            "particles": ["normal", "omit"],
            "register": ["formal", "informal"],
        },
        "style_labels": {
            "directness": {"direct": "직접", "indirect": "간접"},
            "particles": {"normal": "조사", "omit": "생략"},
            "register": {"formal": "존대", "informal": "반말"},
        },
        "destination_seeds": ["강남역", "집"],
        "contact_seeds": ["엄마"],
        "recipient_seeds": ["엄마"],
        "content_seeds": ["곧 도착해"],
        "numeric_samples": {"integer": [0, 22, 100], "number": [16, 22, 30]},
        "unsupported_feature_seeds": ["마사지 시트", "차선 변경"],
        "result_state_seeds": ["춥다", "더워"],
        "danger_vehicle_states": [
            {"label": "driving_forward", "speed_kmh": 60, "gear": "D"},
            {"label": "reversing", "speed_kmh": 5, "gear": "R"},
        ],
        "t5_reason_keywords": ["주행", "위험"],
        "enum_ko_hints": {},
        "output": {"path": "data/drafts/test.jsonl"},
        "model": {"endpoint": "http://localhost:11434"},
        "generator": {"model": "exaone3.5:7.8b", "temperature": 0.4, "endpoint": "http://localhost:11434"},
        "quality_gate": {
            "min_length": 5,
            "max_length": 80,
            "assistant_patterns": ["하겠습니다", "해드리"],
        },
    }


@pytest.fixture()
def registry():
    return load_registry("core8")


def test_enumerate_tool_arg_combos_climate_set(config, registry):
    tool = registry.get_tool("climate_set")
    combos = enumerate_tool_arg_combos(tool, config)
    assert combos
    for args in combos:
        assert any(k in args for k in ("temperature", "fan_level", "mode"))


def test_sample_slots_t1_uses_low_risk_tools(config, registry):
    t1 = sample_slots(registry, config, utterance_type="T1", count=6)
    assert len(t1) == 6
    assert all((tool.risk_tier or 0) <= 1 for tool, _, _ in t1)


def test_sample_compound_slots_t2(config, registry):
    compounds = sample_compound_slots(registry, config, count=4)
    assert len(compounds) == 4
    for calls, order_sensitive, _style in compounds:
        assert len(calls) >= 2
        tool_names = [c["tool"] for c in calls]
        assert len(set(tool_names)) == len(tool_names)
        assert isinstance(order_sensitive, bool)


def test_omit_eligible_slots_skips_default_backed(config, registry):
    tool = registry.get_tool("climate_set")
    args = {"zone": "driver", "temperature": 22}
    eligible = omit_eligible_slots(tool, args)
    assert "zone" not in eligible
    assert "temperature" in eligible


def test_derive_t3_from_t1(config, registry):
    tool = registry.get_tool("window_control")
    base = DraftRecord(
        id="draft_00001",
        utterance_type="T1",
        utterance="운전석 창 열어줘",
        vehicle_state=config["vehicle_state_default"],
        expected=Expected(
            kind="execute",
            calls=[{"tool": "window_control", "args": {"window": "driver", "action": "open"}}],
        ),
        slots={"window": "driver", "action": "open"},
        style=DraftStyle(directness="direct", particles="normal", formality="formal"),
    )
    derived = derive_t3_from_t1(base, tool, __import__("random").Random(0))
    assert derived is not None
    partial, omitted, expected = derived
    assert expected.kind == "clarify"
    assert omitted in expected.missing_slots


def test_make_t5_pairs_execute_and_confirm(config, registry):
    tool = registry.get_tool("trunk_control")
    style = DraftStyle(directness="direct", particles="normal", formality="formal")
    specs = [("트렁크 열어줘", tool, {"action": "open"}, style)] * 2
    pairs = make_t5_pairs_from_utterances(specs, config, min_pairs=2)
    assert len(pairs) >= 2
    safe, danger = pairs[0]
    assert safe.utterance == danger.utterance
    assert safe.expected.kind == "execute"
    assert danger.expected.kind == "confirm"
    assert danger.expected.accept_also == ["refuse"]
    assert safe.state_risk == "safe"
    assert danger.state_risk == "danger"


def test_dedupe_records():
    a = DraftRecord(
        id="1",
        utterance_type="T1",
        utterance="같은 말",
        vehicle_state={},
        expected=Expected(kind="execute", calls=[{"tool": "x", "args": {}}]),
    )
    b = copy.deepcopy(a)
    b.id = "2"
    assert len(dedupe_records([a, b])) == 1


def test_recoverable_numeric_values_filters_keys():
    slots = {"temperature": 22, "zone": "driver", "fan_level": 3}
    assert recoverable_numeric_values(slots) == {22.0, 3.0}


def test_check_unrecoverable_args():
    assert check_unrecoverable_args("온도 24도로", {"temperature": 22})
    assert not check_unrecoverable_args("온도 22도로", {"temperature": 22})
    assert not check_unrecoverable_args("운전석 창 열어줘", {"window": "driver", "action": "open"})


def test_apply_post_filters_adds_unrecoverable_flag():
    record = DraftRecord(
        id="1",
        utterance_type="T1",
        utterance="온도 24도",
        vehicle_state={},
        expected=Expected(kind="execute", calls=[{"tool": "climate_set", "args": {"temperature": 22}}]),
        slots={"temperature": 22},
    )
    filtered = apply_post_filters([record])
    assert "unrecoverable_arg" in filtered[0].flags


def test_validate_utterance_unrecoverable_arg(config):
    assert (
        validate_utterance("좀 세게 틀어줘", config=config, slots={"fan_level": 5}) == "unrecoverable_arg"
    )


def test_dry_run_dataset_invariants(config):
    records, generator = generate_drafts(config, dry_run=True)
    core8 = load_registry("core8")
    core8_names = set(core8.tool_names)

    t2_records = [r for r in records if r.utterance_type == "T2"]
    assert t2_records
    for record in t2_records:
        assert len(record.expected.calls) >= 2
        tools = [c["tool"] for c in record.expected.calls]
        assert len(set(tools)) == len(tools)

    t5_records = [r for r in records if r.utterance_type == "T5"]
    groups: dict[str, list[DraftRecord]] = {}
    for record in t5_records:
        groups.setdefault(record.state_pair_group or "", []).append(record)
    pair_count = sum(1 for g in groups.values() if len(g) == 2)
    assert pair_count >= 2
    for group in groups.values():
        if len(group) != 2:
            continue
        utterances = {r.utterance for r in group}
        kinds = {r.expected.kind for r in group}
        assert len(utterances) == 1
        assert kinds == {"execute", "confirm"}

    tool_names = collect_tool_names_from_records(records)
    assert tool_names.issubset(core8_names)

    for record in records:
        assert record.expected.kind in VALID_EXPECTED_KINDS


def test_parse_utterance_json_extracts_utterance():
    text, reason = parse_utterance_json('{"utterance": "운전석 창 열어줘"}')
    assert text == "운전석 창 열어줘"
    assert reason is None


def test_parse_utterance_json_rejects_plain_text():
    text, reason = parse_utterance_json("그냥 텍스트")
    assert text is None
    assert reason == "json_parse"


def test_validate_utterance_rejects_assistant_voice(config):
    assert validate_utterance("네, 바로 설정하겠습니다", config=config) == "assistant_voice"


def test_validate_utterance_rejects_cjk(config):
    assert validate_utterance("窓を開けて", config=config) == "charset"


def test_validate_utterance_accepts_korean(config):
    assert validate_utterance("운전석 창 좀 열어줘", config=config) is None
