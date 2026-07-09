"""Tests for src.registry: load_registry variants and validate_call error codes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.registry import SCHEMAS_DIR, Registry, load_registry

FULL18_DISTRACTOR_NAMES = {
    "sunroof_control",
    "wiper_control",
    "light_control",
    "drive_mode_set",
    "charge_schedule_set",
    "parking_assist_start",
    "phone_call",
    "message_send",
    "ambient_light_set",
    "defrost_rear",
}


@pytest.fixture()
def core8_registry() -> Registry:
    return load_registry("core8")


@pytest.fixture()
def full18_registry() -> Registry:
    return load_registry("full18")


# ---- load_registry variants ----


def test_load_registry_core8_merges_core_and_tier2(core8_registry: Registry):
    assert set(core8_registry.tool_names) == {
        "climate_set",
        "window_control",
        "navigation_set_destination",
        "media_control",
        "seat_control",
        "vehicle_query",
        "trunk_control",
        "door_lock_control",
    }


def test_load_registry_core8_free_strips_enum_without_mutating_source(core8_registry: Registry):
    free_registry = load_registry("core8_free")

    free_zone = free_registry.get_tool("climate_set").parameters["zone"]
    assert free_zone.enum is None
    assert "driver" in free_zone.description

    core_zone = core8_registry.get_tool("climate_set").parameters["zone"]
    assert core_zone.enum == ["driver", "passenger", "rear", "all"]

    raw = json.loads((SCHEMAS_DIR / "tools_core.json").read_text(encoding="utf-8"))
    zone_raw = next(t for t in raw if t["name"] == "climate_set")["parameters"]["zone"]
    assert zone_raw["enum"] == ["driver", "passenger", "rear", "all"]


def test_load_registry_full18_loads_18_tools(full18_registry: Registry):
    assert len(full18_registry.tool_names) == 18
    assert FULL18_DISTRACTOR_NAMES.issubset(set(full18_registry.tool_names))


def test_load_registry_full18_merges_and_core_wins_dedup(tmp_path: Path):
    for name in ("tools_core.json", "tools_tier2.json"):
        shutil.copy(SCHEMAS_DIR / name, tmp_path / name)

    distractor = [
        {
            "name": "climate_set",
            "risk_tier": 1,
            "description": "Distractor duplicate — must never win over core8.",
            "parameters": {},
            "required": [],
        },
        {
            "name": "sunroof_control",
            "risk_tier": 1,
            "description": "placeholder distractor for dedup test",
            "parameters": {"action": {"type": "string", "enum": ["open", "close"]}},
            "required": ["action"],
        },
    ]
    (tmp_path / "tools_distractor.json").write_text(json.dumps(distractor), encoding="utf-8")

    full = load_registry("full18", schemas_dir=tmp_path)

    assert "sunroof_control" in full.tool_names
    winner = full.get_tool("climate_set")
    assert winner.description == "실내 온도/바람/모드를 설정한다"


def test_full18_message_send_risk_tier_exposed_for_gate(full18_registry: Registry):
    """message_send is in tools_distractor.json but gated via risk_tier metadata."""
    message_send = full18_registry.get_tool("message_send")
    assert message_send is not None
    assert message_send.risk_tier == 2

    tier2_tools = full18_registry.tools_with_risk_tier(2)
    tier2_names = {t.name for t in tier2_tools}
    assert tier2_names == {"trunk_control", "door_lock_control", "message_send"}


def test_load_registry_unknown_variant_raises():
    with pytest.raises(ValueError):
        load_registry("nope")  # type: ignore[arg-type]


# ---- validate_call error codes ----


def test_validate_call_unknown_tool(core8_registry: Registry):
    result = core8_registry.validate_call("teleport", {})
    assert not result.valid
    assert result.error_code == "UNKNOWN_TOOL"


def test_validate_call_missing_required_field(core8_registry: Registry):
    result = core8_registry.validate_call("window_control", {"window": "driver"})
    assert not result.valid
    assert result.error_code == "MISSING_REQUIRED"
    assert result.field == "action"


def test_validate_call_missing_required_one_of(core8_registry: Registry):
    result = core8_registry.validate_call("climate_set", {"zone": "driver"})
    assert not result.valid
    assert result.error_code == "MISSING_REQUIRED"


def test_validate_call_schema_violation_unknown_field(core8_registry: Registry):
    result = core8_registry.validate_call(
        "climate_set", {"zone": "driver", "temperature": 22, "foo": "bar"}
    )
    assert not result.valid
    assert result.error_code == "SCHEMA_VIOLATION"
    assert result.field == "foo"


def test_validate_call_schema_violation_wrong_type(core8_registry: Registry):
    result = core8_registry.validate_call(
        "window_control", {"window": "driver", "action": "set", "level": "fifty"}
    )
    assert not result.valid
    assert result.error_code == "SCHEMA_VIOLATION"
    assert result.field == "level"


def test_validate_call_schema_violation_pattern(full18_registry: Registry):
    bad = full18_registry.validate_call("charge_schedule_set", {"start_time": "25:99"})
    assert not bad.valid
    assert bad.error_code == "SCHEMA_VIOLATION"
    assert bad.field == "start_time"

    ok = full18_registry.validate_call("charge_schedule_set", {"start_time": "07:30"})
    assert ok.valid


def test_validate_call_out_of_range_numeric(core8_registry: Registry):
    result = core8_registry.validate_call("climate_set", {"zone": "driver", "temperature": 99})
    assert not result.valid
    assert result.error_code == "OUT_OF_RANGE"
    assert result.field == "temperature"


def test_validate_call_out_of_range_enum(core8_registry: Registry):
    result = core8_registry.validate_call("vehicle_query", {"item": "engine_temp"})
    assert not result.valid
    assert result.error_code == "OUT_OF_RANGE"
    assert result.field == "item"


def test_validate_call_out_of_range_array_items(core8_registry: Registry):
    result = core8_registry.validate_call(
        "navigation_set_destination", {"destination": "Seoul", "avoid": ["space_travel"]}
    )
    assert not result.valid
    assert result.error_code == "OUT_OF_RANGE"
    assert result.field == "avoid"


def test_validate_call_tier2_tool_without_type_key(core8_registry: Registry):
    ok = core8_registry.validate_call("trunk_control", {"action": "open"})
    assert ok.valid

    bad = core8_registry.validate_call("trunk_control", {"action": "fly"})
    assert not bad.valid
    assert bad.error_code == "OUT_OF_RANGE"
    assert bad.field == "action"


def test_validate_call_valid(core8_registry: Registry):
    result = core8_registry.validate_call(
        "seat_control", {"seat": "driver", "feature": "heat", "level": 2}
    )
    assert result.valid
    assert result.error_code is None
