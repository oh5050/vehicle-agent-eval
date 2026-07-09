"""Tests for review session bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.draft_schema import DraftRecord, Expected
from src.label_schema import record_to_final
from scripts.review import ReviewSession, load_jsonl_records


def test_expected_accept_calls_default():
    exp = Expected(kind="execute", calls=[])
    assert exp.accept_calls == []
    assert exp.accept_also == []


def test_record_to_final_preserves_accept_calls():
    draft = DraftRecord(
        id="draft_00001",
        utterance_type="T6",
        utterance="좀 춥네",
        vehicle_state={"speed_kmh": 0, "gear": "P"},
        expected=Expected(
            kind="execute",
            calls=[{"tool": "climate_set", "args": {"mode": "heat"}}],
            accept_calls=[
                [{"tool": "seat_control", "args": {"seat": "driver", "feature": "heat", "level": 2}}],
            ],
        ),
    )
    final = record_to_final(draft, reviewer="alice")
    assert len(final.expected.accept_calls) == 1
    assert final.expected.accept_calls[0][0]["tool"] == "seat_control"


def _sample_draft() -> DraftRecord:
    return DraftRecord(
        id="draft_00001",
        utterance_type="T1",
        utterance="운전석 온도 22도",
        vehicle_state={
            "speed_kmh": 0,
            "gear": "P",
            "battery_pct": 72,
            "outside_temp_c": 3,
            "passengers": {"front": True, "rear": False},
        },
        expected=Expected(
            kind="execute",
            calls=[{"tool": "climate_set", "args": {"zone": "driver", "temperature": 22}}],
        ),
        slots={"zone": "driver", "temperature": 22},
    )


def test_review_edit_expected_includes_accept_calls(tmp_path: Path):
    draft_path = tmp_path / "empty.jsonl"
    draft_path.write_text("", encoding="utf-8")
    session = ReviewSession(
        draft_path=draft_path,
        final_path=tmp_path / "final.jsonl",
        state_path=tmp_path / ".review.json",
        reviewer="bob",
    )
    current = Expected(
        kind="execute",
        calls=[{"tool": "climate_set", "args": {"mode": "heat"}}],
        accept_calls=[[{"tool": "seat_control", "args": {"seat": "driver", "feature": "heat", "level": 1}}]],
    )
    inputs = iter(["", "", "", "", "", ""])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        updated = session.edit_expected(current)
    assert updated.accept_calls == current.accept_calls
    assert updated.accept_also == []


def test_record_to_final_preserves_expected_shape():
    final = record_to_final(_sample_draft(), reviewer="alice")
    assert final.labeled_by == "alice"
    assert final.expected.kind == "execute"
    assert final.expected.calls[0]["tool"] == "climate_set"
    assert len(final.label_history) == 1


def test_record_to_final_stores_note():
    final = record_to_final(_sample_draft(), reviewer="alice", action="confirm", note="looks good")
    assert final.label_history[0].note == "looks good"


def test_record_to_final_omits_empty_note():
    final = record_to_final(_sample_draft(), reviewer="alice", action="confirm", note=None)
    assert final.label_history[0].note is None


def test_review_session_confirm_with_note(tmp_path: Path):
    draft_path = tmp_path / "sample_draft.jsonl"
    final_path = tmp_path / "sample_final.jsonl"
    draft_path.write_text(json.dumps(_sample_draft().to_jsonl_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    session = ReviewSession(
        draft_path=draft_path,
        final_path=final_path,
        state_path=tmp_path / ".review.json",
        reviewer="bob",
    )
    with patch("builtins.input", return_value="slot mismatch resolved"):
        session.confirm(session.pending[0])

    saved = json.loads(final_path.read_text(encoding="utf-8").strip())
    assert saved["label_history"][0]["note"] == "slot mismatch resolved"


def test_review_session_confirm_skips_empty_note(tmp_path: Path):
    draft_path = tmp_path / "sample_draft.jsonl"
    final_path = tmp_path / "sample_final.jsonl"
    draft_path.write_text(json.dumps(_sample_draft().to_jsonl_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    session = ReviewSession(
        draft_path=draft_path,
        final_path=final_path,
        state_path=tmp_path / ".review.json",
        reviewer="bob",
    )
    with patch("builtins.input", return_value=""):
        session.confirm(session.pending[0])

    saved = json.loads(final_path.read_text(encoding="utf-8").strip())
    assert saved["label_history"][0]["note"] is None


def test_review_session_progress(tmp_path: Path):
    draft_path = tmp_path / "sample_draft.jsonl"
    draft_path.write_text(json.dumps(_sample_draft().to_jsonl_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    session = ReviewSession(
        draft_path=draft_path,
        final_path=tmp_path / "sample_final.jsonl",
        state_path=tmp_path / ".review.json",
        reviewer="bob",
    )
    assert len(session.pending) == 1
    with patch("builtins.input", return_value=""):
        session.confirm(session.pending[0])
    assert len(session.pending) == 0
    assert session.state["confirmed_by_type"]["T1"] == 1


def test_load_jsonl_skips_comments(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text("# comment\n{\"id\":\"1\"}\n", encoding="utf-8")
    rows = load_jsonl_records(path)
    assert rows == [{"id": "1"}]
