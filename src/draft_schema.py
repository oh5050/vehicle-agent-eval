"""Draft and final gold-label record schema (shared expected shape)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UtteranceType = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
ExpectedKind = Literal["execute", "clarify", "refuse", "confirm"]
VALID_EXPECTED_KINDS: frozenset[str] = frozenset({"execute", "clarify", "refuse", "confirm"})


class Expected(BaseModel):
    """Unified expected label — identical in drafts and data/final/."""

    kind: ExpectedKind
    calls: list[dict[str, Any]] = Field(default_factory=list)
    accept_also: list[str] = Field(default_factory=list)
    # Alternative correct call sequences (not kind alternatives — see accept_also).
    accept_calls: list[list[dict[str, Any]]] = Field(default_factory=list)
    reason_keywords: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)


class DraftStyle(BaseModel):
    directness: str
    particles: str
    formality: str


class DraftRecord(BaseModel):
    """One draft dataset row."""

    id: str
    utterance_type: UtteranceType
    utterance: str
    vehicle_state: dict[str, Any]
    expected: Expected
    labeled_by: Literal["llm_draft"] = "llm_draft"
    slots: dict[str, Any] | None = None
    style: DraftStyle | None = None
    order_sensitive: bool | None = None
    omitted_slot: str | None = None
    intended_tool: str | None = None
    result_state: str | None = None
    state_pair_group: str | None = None
    state_risk: Literal["safe", "danger"] | None = None
    flags: list[str] = Field(default_factory=list)
    needs_manual_write: bool = False

    def to_jsonl_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["style"] = self.style.model_dump(mode="json") if self.style else None
        return payload


DRAFT_FILE_HEADER = """\
# WARNING: LLM-generated draft dataset.
# `expected` fields are REFERENCE ONLY — do NOT use for scoring or evaluation
# until a human reviewer moves approved rows to data/final/.
# labeled_by=llm_draft
"""
