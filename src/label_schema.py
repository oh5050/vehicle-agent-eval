"""Final gold-label schema and review history."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.draft_schema import DraftRecord, DraftStyle, Expected, UtteranceType

ReviewAction = Literal["confirm", "modify", "retype", "discard", "manual_add"]


class LabelHistoryEntry(BaseModel):
    at: str
    action: ReviewAction
    reviewer: str
    note: str | None = None
    utterance_type: UtteranceType | None = None
    expected: Expected | None = None


class FinalLabelRecord(BaseModel):
    """One human-confirmed row in data/final/."""

    id: str
    utterance_type: UtteranceType
    utterance: str
    vehicle_state: dict[str, Any]
    expected: Expected
    labeled_by: str = "human"
    label_history: list[LabelHistoryEntry] = Field(default_factory=list)
    slots: dict[str, Any] | None = None
    style: dict[str, Any] | None = None
    order_sensitive: bool | None = None
    omitted_slot: str | None = None
    intended_tool: str | None = None
    result_state: str | None = None
    state_pair_group: str | None = None
    state_risk: Literal["safe", "danger"] | None = None
    flags: list[str] = Field(default_factory=list)
    source_labeled_by: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_to_final(
    record: DraftRecord,
    *,
    reviewer: str,
    action: ReviewAction = "confirm",
    note: str | None = None,
) -> FinalLabelRecord:
    """Copy a draft row to final form — expected schema is already unified."""
    history = [
        LabelHistoryEntry(
            at=utc_now_iso(),
            action=action,
            reviewer=reviewer,
            note=note,
            utterance_type=record.utterance_type,
            expected=record.expected.model_copy(deep=True),
        )
    ]
    return FinalLabelRecord(
        id=record.id,
        utterance_type=record.utterance_type,
        utterance=record.utterance,
        vehicle_state=record.vehicle_state,
        expected=record.expected.model_copy(deep=True),
        labeled_by=reviewer,
        label_history=history,
        slots=record.slots,
        style=record.style.model_dump(mode="json") if record.style else None,
        order_sensitive=record.order_sensitive,
        omitted_slot=record.omitted_slot,
        intended_tool=record.intended_tool,
        result_state=record.result_state,
        state_pair_group=record.state_pair_group,
        state_risk=record.state_risk,
        flags=list(record.flags),
        source_labeled_by=record.labeled_by,
    )
