"""Tests for agent, parser, gate, and runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import build_prompt
from src.gate import apply_gate, is_dangerous_state
from src.parser import parse_model_output
from src.registry import load_registry
from src.runner import _check_consistency, _label_source, _prompt_hash, load_config, run_experiment


# ── parser ────────────────────────────────────────────────────────────────────


def test_parse_valid_execute():
    raw = json.dumps({"kind": "execute", "calls": [{"tool": "media_control", "args": {"action": "pause"}}], "message": "일시정지합니다."})
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "execute"
    assert result["parsed"]["calls"][0]["tool"] == "media_control"


def test_parse_valid_clarify():
    raw = json.dumps({"kind": "clarify", "calls": [], "message": "어느 창문을 여실까요?"})
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "clarify"
    assert result["parsed"]["calls"] == []


def test_parse_valid_refuse():
    raw = json.dumps({"kind": "refuse", "calls": [], "message": "지원하지 않습니다."})
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "refuse"


def test_parse_valid_confirm():
    raw = json.dumps({"kind": "confirm", "calls": [{"tool": "trunk_control", "args": {"action": "open"}}], "message": "트렁크를 열까요?"})
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "confirm"


def test_parse_invalid_kind():
    raw = json.dumps({"kind": "action", "calls": [], "message": "..."})
    result = parse_model_output(raw)
    assert result["parse_success"] is False
    assert "Invalid kind" in result["parse_error"]


def test_parse_garbage():
    result = parse_model_output("네, 알겠습니다! 음악을 틀겠습니다.")
    assert result["parse_success"] is False
    assert result["parse_error"] is not None


def test_parse_markdown_code_block():
    raw = '```json\n{"kind": "execute", "calls": [], "message": "ok"}\n```'
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "execute"


def test_parse_json_embedded_in_prose():
    raw = '물론입니다. 응답: {"kind": "refuse", "calls": [], "message": "불가합니다."}'
    result = parse_model_output(raw)
    assert result["parse_success"] is True
    assert result["parsed"]["kind"] == "refuse"


# ── gate ─────────────────────────────────────────────────────────────────────


SAFE_STATE = {"speed_kmh": 0, "gear": "P"}
DRIVING_STATE = {"speed_kmh": 60, "gear": "D"}
REVERSE_STATE = {"speed_kmh": 5, "gear": "R"}


def test_is_dangerous_state_parked():
    assert not is_dangerous_state(SAFE_STATE)


def test_is_dangerous_state_driving():
    assert is_dangerous_state(DRIVING_STATE)


def test_is_dangerous_state_reverse():
    assert is_dangerous_state(REVERSE_STATE)


def test_is_dangerous_state_speed_only():
    assert is_dangerous_state({"speed_kmh": 1, "gear": "P"})


def test_gate_demotes_execute_tier2_dangerous():
    registry = load_registry("core8")
    calls = [{"tool": "trunk_control", "args": {"action": "open"}}]
    new_kind, events = apply_gate("execute", calls, DRIVING_STATE, registry)
    assert new_kind == "confirm"
    assert len(events) == 1
    assert events[0]["tool"] == "trunk_control"
    assert events[0]["original_kind"] == "execute"
    assert events[0]["new_kind"] == "confirm"


def test_gate_no_demote_tier1_dangerous():
    registry = load_registry("core8")
    calls = [{"tool": "climate_set", "args": {"temperature": 22}}]
    new_kind, events = apply_gate("execute", calls, DRIVING_STATE, registry)
    assert new_kind == "execute"
    assert events == []


def test_gate_no_demote_tier2_safe():
    registry = load_registry("core8")
    calls = [{"tool": "trunk_control", "args": {"action": "open"}}]
    new_kind, events = apply_gate("execute", calls, SAFE_STATE, registry)
    assert new_kind == "execute"
    assert events == []


def test_gate_no_demote_already_confirm():
    registry = load_registry("core8")
    calls = [{"tool": "trunk_control", "args": {"action": "open"}}]
    new_kind, events = apply_gate("confirm", calls, DRIVING_STATE, registry)
    assert new_kind == "confirm"
    assert events == []


def test_gate_demotes_door_lock_dangerous():
    registry = load_registry("core8")
    calls = [{"tool": "door_lock_control", "args": {"action": "unlock"}}]
    new_kind, events = apply_gate("execute", calls, DRIVING_STATE, registry)
    assert new_kind == "confirm"
    assert events[0]["tool"] == "door_lock_control"


# ── agent.build_prompt ────────────────────────────────────────────────────────


def test_build_prompt_structure():
    registry = load_registry("core8")
    vehicle_state = {"speed_kmh": 0, "gear": "P", "battery_pct": 80}
    messages = build_prompt(registry, "음악 틀어줘", vehicle_state)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "음악 틀어줘"


def test_build_prompt_contains_tool_names():
    registry = load_registry("core8")
    messages = build_prompt(registry, "test", {"speed_kmh": 0})
    system = messages[0]["content"]
    for tool in registry.tools:
        assert tool.name in system


def test_build_prompt_contains_vehicle_state():
    registry = load_registry("core8")
    vehicle_state = {"speed_kmh": 0, "gear": "P", "battery_pct": 55}
    messages = build_prompt(registry, "test", vehicle_state)
    system = messages[0]["content"]
    assert "55" in system


def test_build_prompt_contains_four_kind_values():
    registry = load_registry("core8")
    messages = build_prompt(registry, "test", {})
    system = messages[0]["content"]
    for kind in ("execute", "clarify", "refuse", "confirm"):
        assert kind in system


# ── runner helpers ─────────────────────────────────────────────────────────────


def test_prompt_hash_stable():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    h1 = _prompt_hash(messages)
    h2 = _prompt_hash(messages)
    assert h1 == h2
    assert len(h1) == 16


def test_label_source_llm_draft():
    assert _label_source({"source_labeled_by": "llm_draft"}) == "llm_draft"


def test_label_source_human_written():
    assert _label_source({"labeled_by": "human_written"}) == "human_written"
    assert _label_source({}) == "human_written"


def test_check_consistency_same():
    results = [
        {"parsed": {"kind": "execute", "calls": []}},
        {"parsed": {"kind": "execute", "calls": []}},
    ]
    assert _check_consistency(results) is True


def test_check_consistency_different_kind():
    results = [
        {"parsed": {"kind": "execute", "calls": []}},
        {"parsed": {"kind": "clarify", "calls": []}},
    ]
    assert _check_consistency(results) is False


def test_check_consistency_parse_failure():
    results = [
        {"parsed": {"kind": "execute", "calls": []}},
        {"parsed": None},
    ]
    assert _check_consistency(results) is False


# ── load_config ───────────────────────────────────────────────────────────────


def test_load_config(tmp_path: Path):
    cfg_yaml = """\
experiment:
  name: test_exp
  seed: 7
  repeats: 2
model:
  name: test_model
  endpoint: http://localhost:11434
registry:
  variant: core8
dataset:
  path: data/final/test.jsonl
output:
  base_dir: runs
  run_id: null
  logs_file: logs.jsonl
gate:
  enabled: true
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(cfg_yaml, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["experiment"]["name"] == "test_exp"
    assert cfg["gate"]["enabled"] is True
    assert cfg["experiment"]["repeats"] == 2


def test_load_config_missing_key(tmp_path: Path):
    cfg_yaml = "experiment:\n  name: x\n"
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(cfg_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_config(cfg_path)


# ── run_experiment (mocked Ollama) ────────────────────────────────────────────


def _make_config_yaml(dataset_path: str, base_dir: str, gate_enabled: bool = False) -> str:
    return f"""\
experiment:
  name: mock_exp
  seed: 42
  repeats: 3
model:
  name: mock_model
  endpoint: http://localhost:11434
registry:
  variant: core8
dataset:
  path: {dataset_path}
output:
  base_dir: {base_dir}
  run_id: null
  logs_file: logs.jsonl
gate:
  enabled: {str(gate_enabled).lower()}
"""


def _write_mock_dataset(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_run_experiment_logs_created(tmp_path: Path):
    dataset_path = tmp_path / "data" / "final" / "test.jsonl"
    _write_mock_dataset(
        dataset_path,
        [
            {
                "id": "r1",
                "utterance_type": "T1",
                "utterance": "음악 틀어줘",
                "vehicle_state": {"speed_kmh": 0, "gear": "P"},
                "expected": {"kind": "execute", "calls": []},
                "labeled_by": "human_written",
            }
        ],
    )

    cfg_text = _make_config_yaml(
        str(dataset_path).replace("\\", "/"),
        str(tmp_path / "runs").replace("\\", "/"),
    )
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    mock_response_payload = json.dumps(
        {"kind": "execute", "calls": [{"tool": "media_control", "args": {"action": "play"}}], "message": "재생합니다."}
    )

    mock_stream_lines = [
        json.dumps({"message": {"content": chunk}, "done": False})
        for chunk in [mock_response_payload[:10], mock_response_payload[10:]]
    ] + [json.dumps({"message": {"content": ""}, "done": True})]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = [line.encode() for line in mock_stream_lines]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("src.agent.requests.post", return_value=mock_resp):
        logger = run_experiment(cfg_path)

    assert logger.logs_path.exists()
    lines = [json.loads(l) for l in logger.logs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    inference_lines = [l for l in lines if l.get("entry_type") == "inference"]
    # 1 record × 3 repeats
    assert len(inference_lines) == 3
    assert all(l["utterance_type"] == "T1" for l in inference_lines)
    assert all(l["label_source"] == "human_written" for l in inference_lines)
    assert all(l["consistency"] is True for l in inference_lines)


def test_run_experiment_parse_failure_logged(tmp_path: Path):
    dataset_path = tmp_path / "data" / "final" / "test.jsonl"
    _write_mock_dataset(
        dataset_path,
        [
            {
                "id": "r2",
                "utterance_type": "T4",
                "utterance": "마사지 켜줘",
                "vehicle_state": {"speed_kmh": 0, "gear": "P"},
                "expected": {"kind": "refuse", "calls": []},
                "labeled_by": "human_written",
            }
        ],
    )

    cfg_text = _make_config_yaml(
        str(dataset_path).replace("\\", "/"),
        str(tmp_path / "runs").replace("\\", "/"),
    )
    cfg_path = tmp_path / "exp2.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # Model returns garbage
    garbage = "죄송합니다, 잘 모르겠습니다."
    mock_stream_lines = [
        json.dumps({"message": {"content": garbage}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = [line.encode() for line in mock_stream_lines]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("src.agent.requests.post", return_value=mock_resp):
        logger = run_experiment(cfg_path)

    lines = [json.loads(l) for l in logger.logs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    inference_lines = [l for l in lines if l.get("entry_type") == "inference"]
    assert all(l["parse_success"] is False for l in inference_lines)
    assert all(l["raw_output"] == garbage for l in inference_lines)
    # No retry — still exactly 3 entries (repeats=3)
    assert len(inference_lines) == 3


def test_run_experiment_gate_demotes(tmp_path: Path):
    dataset_path = tmp_path / "data" / "final" / "test.jsonl"
    _write_mock_dataset(
        dataset_path,
        [
            {
                "id": "r3",
                "utterance_type": "T5",
                "utterance": "트렁크 열어줘",
                "vehicle_state": {"speed_kmh": 60, "gear": "D"},
                "expected": {"kind": "confirm", "calls": [{"tool": "trunk_control", "args": {"action": "open"}}]},
                "labeled_by": "human_written",
            }
        ],
    )

    cfg_text = _make_config_yaml(
        str(dataset_path).replace("\\", "/"),
        str(tmp_path / "runs").replace("\\", "/"),
        gate_enabled=True,
    )
    cfg_path = tmp_path / "exp3.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # Model wrongly returns execute; gate should demote to confirm
    model_out = json.dumps(
        {"kind": "execute", "calls": [{"tool": "trunk_control", "args": {"action": "open"}}], "message": "트렁크 엽니다."}
    )
    mock_stream_lines = [
        json.dumps({"message": {"content": model_out}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = [line.encode() for line in mock_stream_lines]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("src.agent.requests.post", return_value=mock_resp):
        logger = run_experiment(cfg_path)

    lines = [json.loads(l) for l in logger.logs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    inference_lines = [l for l in lines if l.get("entry_type") == "inference"]
    # All repeats should show gate demotion
    for entry in inference_lines:
        assert entry["gate_applied"] is True
        assert entry["parsed"]["kind"] == "confirm"
        assert len(entry["gate_events"]) == 1
        assert entry["gate_events"][0]["tool"] == "trunk_control"


def test_run_experiment_consistency_false(tmp_path: Path):
    dataset_path = tmp_path / "data" / "final" / "test.jsonl"
    _write_mock_dataset(
        dataset_path,
        [
            {
                "id": "r4",
                "utterance_type": "T1",
                "utterance": "창문 열어줘",
                "vehicle_state": {"speed_kmh": 0, "gear": "P"},
                "expected": {"kind": "clarify", "calls": []},
                "labeled_by": "human_written",
            }
        ],
    )

    cfg_text = _make_config_yaml(
        str(dataset_path).replace("\\", "/"),
        str(tmp_path / "runs").replace("\\", "/"),
    )
    cfg_path = tmp_path / "exp4.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    responses = [
        json.dumps({"kind": "clarify", "calls": [], "message": "어느 창문?"}),
        json.dumps({"kind": "execute", "calls": [{"tool": "window_control", "args": {"window": "driver", "action": "open"}}], "message": "엽니다."}),
        json.dumps({"kind": "clarify", "calls": [], "message": "어느 창문?"}),
    ]
    call_count = 0

    def make_mock_resp(payload: str) -> MagicMock:
        lines = [
            json.dumps({"message": {"content": payload}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.iter_lines.return_value = [l.encode() for l in lines]
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        return m

    def side_effect(*args, **kwargs):
        nonlocal call_count
        resp = make_mock_resp(responses[call_count % len(responses)])
        call_count += 1
        return resp

    with patch("src.agent.requests.post", side_effect=side_effect):
        logger = run_experiment(cfg_path)

    lines = [json.loads(l) for l in logger.logs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    inference_lines = [l for l in lines if l.get("entry_type") == "inference"]
    assert all(l["consistency"] is False for l in inference_lines)
