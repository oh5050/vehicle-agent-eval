#!/usr/bin/env python3
"""Compare utterance generators across Ollama models on fixed slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_matrix import (  # noqa: E402
    UtteranceGenerator,
    build_compound_utterance_prompt,
    build_t6_prompt,
    build_utterance_prompt,
    sample_compound_slots,
    sample_slots,
    slots_to_prompt_payload,
)
from src.registry import load_registry

DEFAULT_MODELS = ("exaone3.5:7.8b", "qwen2.5:7b")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fixed_slot_jobs(registry, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return 6 jobs: T1×2, T2×2, T6×2 with deterministic sampling."""
    compare_cfg = dict(config)
    compare_cfg["types"] = {
        **compare_cfg.get("types", {}),
        "T1": {"count": 2, "max_risk_tier": 1},
        "T2": {"count": 2, "max_risk_tier": 1},
    }
    jobs: list[dict[str, Any]] = []

    for tool, args, style in sample_slots(registry, compare_cfg, utterance_type="T1", count=2):
        slot_payload = slots_to_prompt_payload(tool, args, config)
        jobs.append(
            {
                "label": f"T1/{tool.name}",
                "type": "T1",
                "slots": args,
                "prompt": build_utterance_prompt(
                    task="슬롯 값을 모두 반영한 자연스러운 차량 음성 명령 발화를 작성하세요.",
                    slot_payload=slot_payload,
                    style=style,
                    style_labels=config["style_labels"],
                ),
                "meta": {"calls": [{"tool": tool.name, "args": args}]},
            }
        )

    for calls, order_sensitive, style in sample_compound_slots(registry, compare_cfg, count=2):
        tool_names = "+".join(c["tool"] for c in calls)
        jobs.append(
            {
                "label": f"T2/{tool_names}",
                "type": "T2",
                "slots": {"calls": calls, "order_sensitive": order_sensitive},
                "prompt": build_compound_utterance_prompt(
                    calls=calls,
                    registry=registry,
                    config=config,
                    style=style,
                    style_labels=config["style_labels"],
                    order_sensitive=order_sensitive,
                ),
                "meta": {"calls": calls, "order_sensitive": order_sensitive},
            }
        )

    seeds = config.get("result_state_seeds", ["춥다", "졸리다"])
    from src.draft_schema import DraftStyle

    for i, result_state in enumerate(seeds[:2]):
        style = DraftStyle(directness="direct", particles="normal", formality="formal")
        jobs.append(
            {
                "label": f"T6/{result_state}",
                "type": "T6",
                "prompt": build_t6_prompt(result_state, style, config["style_labels"]),
                "meta": {"result_state": result_state},
            }
        )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B compare utterance generators.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gen_drafts.yaml")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry("core8")
    jobs = fixed_slot_jobs(registry, config)

    print("=" * 100)
    print("Utterance generator A/B — fixed slots: T1×2, T2×2, T6×2")
    print("=" * 100)

    for job_idx, job in enumerate(jobs):
        print(f"\n[{job['label']}] meta={json.dumps(job['meta'], ensure_ascii=False)}")
        print("-" * 100)
        for model in args.models:
            generator = UtteranceGenerator(config, model=model)
            result = generator.generate(job["prompt"], seed=args.seed + job_idx, slots=job.get("slots"))
            utterance = result.utterance or "(needs_manual_write)"
            manual = " [MANUAL]" if result.needs_manual_write else ""
            print(f"  {model:20} | reject={generator.stats.reject_rate:5.1%} | {utterance}{manual}")
            if generator.stats.rejects:
                reasons = ", ".join(f"{k}:{v}" for k, v in sorted(generator.stats.rejects.items()))
                print(f"  {'':20}   reasons: {reasons}")

    print("\n" + "=" * 100)
    print("Summary reject rates:")
    for model in args.models:
        generator = UtteranceGenerator(config, model=model)
        for job_idx, job in enumerate(jobs):
            generator.generate(job["prompt"], seed=args.seed + job_idx)
        print(f"  {model:20} {generator.stats.summary_lines()[0]}")


if __name__ == "__main__":
    main()
