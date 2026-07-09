#!/usr/bin/env python3
"""Interactive CLI to review data/drafts/ one record at a time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_schema import DraftRecord, Expected, UtteranceType  # noqa: E402
from src.label_schema import FinalLabelRecord, record_to_final  # noqa: E402

UTTERANCE_TYPES: tuple[UtteranceType, ...] = ("T1", "T2", "T3", "T4", "T5", "T6")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        records.append(json.loads(stripped))
    return records


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"confirmed_ids": [], "discarded_ids": [], "confirmed_by_type": {t: 0 for t in UTTERANCE_TYPES}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class ReviewSession:
    def __init__(self, *, draft_path: Path, final_path: Path, state_path: Path, reviewer: str) -> None:
        self.draft_path = draft_path
        self.final_path = final_path
        self.state_path = state_path
        self.reviewer = reviewer
        self.drafts = [DraftRecord.model_validate(r) for r in load_jsonl_records(draft_path)]
        self.state = load_state(state_path)
        if "confirmed_by_type" not in self.state:
            self.state["confirmed_by_type"] = {t: 0 for t in UTTERANCE_TYPES}

    @property
    def processed_ids(self) -> set[str]:
        return set(self.state["confirmed_ids"]) | set(self.state["discarded_ids"])

    @property
    def pending(self) -> list[DraftRecord]:
        return [d for d in self.drafts if d.id not in self.processed_ids]

    def progress_line(self) -> str:
        total = len(self.drafts)
        done = len(self.processed_ids)
        confirmed = len(self.state["confirmed_ids"])
        discarded = len(self.state["discarded_ids"])
        remaining = total - done
        pct = (done / total * 100) if total else 100.0
        by_type = " | ".join(f"{t}:{self.state['confirmed_by_type'].get(t, 0)}" for t in UTTERANCE_TYPES)
        return (
            f"진행 {done}/{total} ({pct:.1f}%) | 남음 {remaining} | "
            f"확정 {confirmed} | 폐기 {discarded} || 유형별 확정: {by_type}"
        )

    def show_record(self, record: DraftRecord) -> None:
        exp = record.expected
        print("\n" + "=" * 72)
        print(self.progress_line())
        print("=" * 72)
        print(f"ID       : {record.id}")
        print(f"유형     : {record.utterance_type}")
        print(f"발화     : {record.utterance}")
        if record.flags:
            print(f"플래그   : {', '.join(record.flags)}")
        if record.slots:
            print(f"슬롯     : {json.dumps(record.slots, ensure_ascii=False)}")
        if record.order_sensitive is not None:
            print(f"순서민감 : {record.order_sensitive}")
        print(f"차량상태 : {json.dumps(record.vehicle_state, ensure_ascii=False)}")
        print("expected :")
        print(f"  kind            : {exp.kind}")
        print(f"  calls           : {json.dumps(exp.calls, ensure_ascii=False)}")
        print(f"  accept_also     : {json.dumps(exp.accept_also, ensure_ascii=False)}")
        print(f"  accept_calls    : {json.dumps(exp.accept_calls, ensure_ascii=False)}")
        print(f"  reason_keywords : {json.dumps(exp.reason_keywords, ensure_ascii=False)}")
        print(f"  missing_slots   : {json.dumps(exp.missing_slots, ensure_ascii=False)}")
        if record.omitted_slot:
            print(f"생략슬롯 : {record.omitted_slot}")
        if record.result_state:
            print(f"결과상태 : {record.result_state}")
        if record.state_pair_group:
            print(f"T5쌍     : {record.state_pair_group} ({record.state_risk})")
        print("-" * 72)
        print("[1] 확정  [2] 수정  [3] 유형변경  [4] 폐기  [q] 종료")

    def _mark_confirmed(self, record_id: str, utterance_type: UtteranceType) -> None:
        if record_id not in self.state["confirmed_ids"]:
            self.state["confirmed_ids"].append(record_id)
        self.state["confirmed_by_type"][utterance_type] = self.state["confirmed_by_type"].get(utterance_type, 0) + 1

    def _mark_discarded(self, record_id: str) -> None:
        if record_id not in self.state["discarded_ids"]:
            self.state["discarded_ids"].append(record_id)

    def _save_final(self, final: FinalLabelRecord) -> None:
        append_jsonl(self.final_path, final.to_jsonl_dict())
        self._mark_confirmed(final.id, final.utterance_type)
        save_state(self.state_path, self.state)

    def prompt_note(self, *, action: str) -> str | None:
        if action == "modify":
            print("메모(권장): 무엇을 왜 바꿨는지 한 줄로 적어주세요. (Enter = 생략)")
        else:
            print("메모(선택): 한 줄 메모 — Enter로 생략")
        value = input("note> ").strip()
        return value or None

    def confirm(self, record: DraftRecord) -> None:
        note = self.prompt_note(action="confirm")
        final = record_to_final(record, reviewer=self.reviewer, action="confirm", note=note)
        self._save_final(final)
        print(f"✓ 확정 → {self.final_path}")

    def discard(self, record: DraftRecord) -> None:
        self._mark_discarded(record.id)
        save_state(self.state_path, self.state)
        print("✗ 폐기")

    def prompt_input(self, label: str, current: str) -> str:
        value = input(f"{label} [{current}]: ").strip()
        return value if value else current

    def prompt_json(self, label: str, current: Any) -> Any:
        rendered = json.dumps(current, ensure_ascii=False)
        value = input(f"{label} (JSON) [{rendered}]: ").strip()
        return json.loads(value) if value else current

    def edit_expected(self, current: Expected) -> Expected:
        kind = self.prompt_input("expected.kind", current.kind)
        calls = self.prompt_json("expected.calls", current.calls)
        accept_also = self.prompt_json("expected.accept_also", current.accept_also)
        accept_calls = self.prompt_json("expected.accept_calls", current.accept_calls)
        reason_keywords = self.prompt_json("expected.reason_keywords", current.reason_keywords)
        missing_slots = self.prompt_json("expected.missing_slots", current.missing_slots)
        return Expected(
            kind=kind,  # type: ignore[arg-type]
            calls=calls,
            accept_also=accept_also,
            accept_calls=accept_calls,
            reason_keywords=reason_keywords,
            missing_slots=missing_slots,
        )

    def edit_record(self, record: DraftRecord) -> None:
        updated = record.model_copy(deep=True)
        updated.utterance = self.prompt_input("발화", record.utterance)
        updated.expected = self.edit_expected(record.expected)
        note = self.prompt_note(action="modify")
        final = record_to_final(updated, reviewer=self.reviewer, action="modify", note=note)
        self._save_final(final)
        print(f"✓ 수정 후 확정 → {self.final_path}")

    def retype(self, record: DraftRecord) -> None:
        print(f"현재 유형: {record.utterance_type}")
        new_type = input(f"새 유형 ({'/'.join(UTTERANCE_TYPES)}): ").strip().upper()
        if new_type not in UTTERANCE_TYPES:
            print("유효하지 않은 유형 — 건너뜀")
            return
        updated = record.model_copy(deep=True)
        updated.utterance_type = new_type  # type: ignore[assignment]
        note = self.prompt_note(action="retype")
        final = record_to_final(updated, reviewer=self.reviewer, action="retype", note=note)
        if note is None:
            final.label_history[0].note = f"{record.utterance_type} -> {new_type}"
        self._save_final(final)
        print(f"✓ 유형변경({new_type}) 후 확정 → {self.final_path}")

    def run(self) -> None:
        if not self.drafts:
            print(f"검수할 draft가 없습니다: {self.draft_path}")
            return

        print(f"Draft  : {self.draft_path}")
        print(f"Final  : {self.final_path}")
        print(f"State  : {self.state_path}")

        while self.pending:
            record = self.pending[0]
            self.show_record(record)
            choice = input("선택> ").strip().lower()

            if choice in ("q", "quit", "exit"):
                print("검수 종료")
                break
            if choice in ("1", "확정"):
                self.confirm(record)
            elif choice in ("2", "수정"):
                self.edit_record(record)
            elif choice in ("3", "유형변경"):
                self.retype(record)
            elif choice in ("4", "폐기"):
                self.discard(record)
            else:
                print("잘못된 입력 — 1/2/3/4/q")

        if not self.pending:
            print("\n모든 draft 검수 완료")
            print(self.progress_line())


def default_final_path(draft_path: Path) -> Path:
    stem = draft_path.stem.replace("_draft", "")
    return ROOT / "data" / "final" / f"{stem}_final.jsonl"


def default_state_path(draft_path: Path) -> Path:
    return draft_path.parent / f".review_{draft_path.stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review draft JSONL records interactively.")
    parser.add_argument("--draft", type=Path, default=ROOT / "data" / "drafts" / "dataset_draft.jsonl")
    parser.add_argument("--final", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--reviewer", default="human")
    args = parser.parse_args()

    draft_path = args.draft if args.draft.is_absolute() else ROOT / args.draft
    session = ReviewSession(
        draft_path=draft_path,
        final_path=args.final or default_final_path(draft_path),
        state_path=args.state or default_state_path(draft_path),
        reviewer=args.reviewer,
    )
    session.run()


if __name__ == "__main__":
    main()
