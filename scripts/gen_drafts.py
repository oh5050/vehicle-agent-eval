#!/usr/bin/env python3
"""Generate matrix-sampled draft dataset rows under data/drafts/."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_matrix import (  # noqa: E402
    UtteranceGenerator,
    apply_post_filters,
    build_clarify_expected,
    build_compound_utterance_prompt,
    build_execute_expected,
    build_refuse_expected,
    build_t4_prompt,
    build_t6_prompt,
    build_utterance_prompt,
    call_entry,
    derive_t3_from_t1,
    make_t5_pairs_from_utterances,
    sample_compound_slots,
    sample_slots,
    sample_tier2_single_slots,
    slots_to_prompt_payload,
    vary_unsupported_feature,
)
from src.draft_schema import DRAFT_FILE_HEADER, DraftRecord, DraftStyle  # noqa: E402
from src.registry import Registry, ToolSchema, load_registry  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def tool_by_name(registry: Registry, name: str) -> ToolSchema:
    tool = registry.get_tool(name)
    if tool is None:
        raise KeyError(name)
    return tool


def _attach_utterance(
    record: DraftRecord,
    generator: UtteranceGenerator,
    prompt: str,
    seed: int,
    *,
    slots: dict[str, Any] | None = None,
) -> None:
    slot_ctx = slots if slots is not None else record.slots
    result = generator.generate(prompt, seed=seed, slots=slot_ctx)
    record.utterance = result.utterance
    record.needs_manual_write = result.needs_manual_write
    if result.needs_manual_write and "needs_manual_write" not in record.flags:
        record.flags.append("needs_manual_write")


def generate_t1(
    registry: Registry,
    config: dict[str, Any],
    generator: UtteranceGenerator,
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    count = config["types"]["T1"]["count"]
    if count <= 0:
        return [], id_counter
    default_state = copy.deepcopy(config["vehicle_state_default"])
    style_labels = config["style_labels"]
    records: list[DraftRecord] = []

    for tool, args, style in sample_slots(registry, config, utterance_type="T1", count=count):
        slot_payload = slots_to_prompt_payload(tool, args, config)
        prompt = build_utterance_prompt(
            task="슬롯 값을 모두 반영한 자연스러운 차량 음성 명령 발화를 작성하세요.",
            slot_payload=slot_payload,
            style=style,
            style_labels=style_labels,
        )
        calls = [call_entry(tool.name, args)]
        record = DraftRecord(
            id=f"draft_{id_counter:05d}",
            utterance_type="T1",
            utterance="",
            vehicle_state=copy.deepcopy(default_state),
            expected=build_execute_expected(calls),
            slots=copy.deepcopy(args),
            style=style,
        )
        _attach_utterance(record, generator, prompt, config["seed"] + id_counter)
        records.append(record)
        id_counter += 1
    return records, id_counter


def generate_t2(
    registry: Registry,
    config: dict[str, Any],
    generator: UtteranceGenerator,
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    count = config["types"]["T2"]["count"]
    if count <= 0:
        return [], id_counter
    default_state = copy.deepcopy(config["vehicle_state_default"])
    style_labels = config["style_labels"]
    records: list[DraftRecord] = []

    for calls, order_sensitive, style in sample_compound_slots(registry, config, count=count):
        prompt = build_compound_utterance_prompt(
            calls=calls,
            registry=registry,
            config=config,
            style=style,
            style_labels=style_labels,
            order_sensitive=order_sensitive,
        )
        record = DraftRecord(
            id=f"draft_{id_counter:05d}",
            utterance_type="T2",
            utterance="",
            vehicle_state=copy.deepcopy(default_state),
            expected=build_execute_expected(calls),
            slots={"calls": copy.deepcopy(calls), "order_sensitive": order_sensitive},
            style=style,
            order_sensitive=order_sensitive,
        )
        _attach_utterance(record, generator, prompt, config["seed"] + id_counter)
        records.append(record)
        id_counter += 1
    return records, id_counter


def generate_t3(
    registry: Registry,
    config: dict[str, Any],
    generator: UtteranceGenerator,
    t1_records: list[DraftRecord],
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    count = config["types"]["T3"]["count"]
    if count <= 0:
        return [], id_counter
    rng = random.Random(config["seed"] + 303)
    style_labels = config["style_labels"]
    default_state = copy.deepcopy(config["vehicle_state_default"])
    records: list[DraftRecord] = []

    candidates = [r for r in t1_records if r.expected.calls and not r.needs_manual_write]
    rng.shuffle(candidates)

    for base in candidates:
        if len(records) >= count:
            break
        tool_name = base.expected.calls[0]["tool"]
        tool = tool_by_name(registry, tool_name)
        derived = derive_t3_from_t1(base, tool, rng)
        if derived is None:
            continue
        partial, omitted, expected = derived
        style = base.style or DraftStyle(directness="direct", particles="normal", formality="formal")
        slot_payload = slots_to_prompt_payload(tool, partial, config)
        prompt = build_utterance_prompt(
            task="슬롯 중 일부가 빠진 불완전 발화를 작성하세요. 빠진 정보는 언급하지 마세요.",
            slot_payload=slot_payload,
            style=style,
            style_labels=style_labels,
            extra_instructions=f"의도적으로 생략된 슬롯: {omitted}",
        )
        record = DraftRecord(
            id=f"draft_{id_counter:05d}",
            utterance_type="T3",
            utterance="",
            vehicle_state=copy.deepcopy(default_state),
            expected=expected,
            slots=partial,
            style=style,
            omitted_slot=omitted,
        )
        _attach_utterance(record, generator, prompt, config["seed"] + id_counter)
        records.append(record)
        id_counter += 1

    if len(records) < count:
        raise ValueError(f"T3 generation produced {len(records)} rows, expected {count}")
    return records, id_counter


def generate_t4(
    config: dict[str, Any],
    generator: UtteranceGenerator,
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    count = config["types"]["T4"]["count"]
    if count <= 0:
        return [], id_counter
    rng = random.Random(config["seed"] + 404)
    style_labels = config["style_labels"]
    default_state = copy.deepcopy(config["vehicle_state_default"])
    seeds = config["unsupported_feature_seeds"]
    records: list[DraftRecord] = []

    for _ in range(count):
        feature = vary_unsupported_feature(rng.choice(seeds), rng)
        style = DraftStyle(
            directness=rng.choice(config["styles"]["directness"]),
            particles=rng.choice(config["styles"]["particles"]),
            formality=rng.choice(config["styles"]["register"]),
        )
        prompt = build_t4_prompt(feature, style, style_labels)
        record = DraftRecord(
            id=f"draft_{id_counter:05d}",
            utterance_type="T4",
            utterance="",
            vehicle_state=copy.deepcopy(default_state),
            expected=build_refuse_expected(reason_keywords=[feature]),
            style=style,
        )
        _attach_utterance(record, generator, prompt, config["seed"] + id_counter)
        records.append(record)
        id_counter += 1
    return records, id_counter


def generate_t5(
    registry: Registry,
    config: dict[str, Any],
    generator: UtteranceGenerator,
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    min_pairs = config["types"]["T5"]["min_state_pairs"]
    if min_pairs <= 0:
        return [], id_counter
    style_labels = config["style_labels"]
    utterance_specs: list[tuple[str, ToolSchema, dict[str, Any], DraftStyle]] = []

    for tool, args, style in sample_tier2_single_slots(registry, config, count=min_pairs):
        slot_payload = slots_to_prompt_payload(tool, args, config)
        prompt = build_utterance_prompt(
            task="tier-2 툴 슬롯을 반영한 정상 차량 음성 명령 발화를 작성하세요.",
            slot_payload=slot_payload,
            style=style,
            style_labels=style_labels,
        )
        result = generator.generate(prompt, seed=config["seed"] + id_counter, slots=args)
        if result.needs_manual_write:
            id_counter += 1
            continue
        utterance_specs.append((result.utterance, tool, args, style))
        id_counter += 1

    pairs = make_t5_pairs_from_utterances(utterance_specs, config, min_pairs=min_pairs)
    records = [record for pair in pairs for record in pair]
    return records, id_counter


def generate_t6(
    config: dict[str, Any],
    generator: UtteranceGenerator,
    *,
    id_counter: int,
) -> tuple[list[DraftRecord], int]:
    count = config["types"]["T6"]["count"]
    if count <= 0:
        return [], id_counter
    rng = random.Random(config["seed"] + 606)
    style_labels = config["style_labels"]
    default_state = copy.deepcopy(config["vehicle_state_default"])
    seeds = config["result_state_seeds"]
    records: list[DraftRecord] = []

    for i in range(count):
        result_state = seeds[i % len(seeds)]
        style = DraftStyle(
            directness=rng.choice(config["styles"]["directness"]),
            particles=rng.choice(config["styles"]["particles"]),
            formality=rng.choice(config["styles"]["register"]),
        )
        prompt = build_t6_prompt(result_state, style, style_labels)
        record = DraftRecord(
            id=f"draft_{id_counter:05d}",
            utterance_type="T6",
            utterance="",
            vehicle_state=copy.deepcopy(default_state),
            expected=build_execute_expected([]),
            result_state=result_state,
            style=style,
        )
        _attach_utterance(record, generator, prompt, config["seed"] + id_counter)
        records.append(record)
        id_counter += 1
    return records, id_counter


def write_jsonl(path: Path, records: list[DraftRecord]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(DRAFT_FILE_HEADER)
        for record in records:
            f.write(json.dumps(record.to_jsonl_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")


def generate_drafts(config: dict[str, Any], *, dry_run: bool = False) -> tuple[list[DraftRecord], UtteranceGenerator]:
    if config["registry"]["variant"] != "core8":
        raise ValueError("registry.variant must be core8 — gold labels must not depend on distractor tools")

    registry = load_registry("core8")
    generator = UtteranceGenerator(config, dry_run=dry_run)
    id_counter = 1
    all_records: list[DraftRecord] = []

    t1_records, id_counter = generate_t1(registry, config, generator, id_counter=id_counter)
    all_records.extend(t1_records)

    t2_records, id_counter = generate_t2(registry, config, generator, id_counter=id_counter)
    all_records.extend(t2_records)

    t3_records, id_counter = generate_t3(registry, config, generator, t1_records, id_counter=id_counter)
    all_records.extend(t3_records)

    t4_records, id_counter = generate_t4(config, generator, id_counter=id_counter)
    all_records.extend(t4_records)

    t5_records, id_counter = generate_t5(registry, config, generator, id_counter=id_counter)
    all_records.extend(t5_records)

    t6_records, id_counter = generate_t6(config, generator, id_counter=id_counter)
    all_records.extend(t6_records)

    return apply_post_filters(all_records), generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matrix-sampled draft dataset via Ollama.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gen_drafts.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Skip Ollama calls; emit placeholder utterances.")
    args = parser.parse_args()

    config = load_config(args.config)
    records, generator = generate_drafts(config, dry_run=args.dry_run)
    output_path = ROOT / config["output"]["path"]
    write_jsonl(output_path, records)

    flagged = sum(1 for r in records if r.flags)
    manual = sum(1 for r in records if r.needs_manual_write)
    print(f"Wrote {len(records)} draft rows to {output_path} ({flagged} flagged, {manual} needs_manual_write)")
    print(f"Generator: {generator.settings['model']} (temp={generator.settings['temperature']})")
    for line in generator.stats.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
