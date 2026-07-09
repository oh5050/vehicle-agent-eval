"""Experiment runner: config-driven evaluation loop.

Flow:
    load_config → load_registry → load_dataset
    for each record × repeats:
        build_prompt → call_model → parse_output → apply_gate (if enabled)
        collect per-repeat result
    compute consistency across repeats
    log all repeats to runs/{run_id}/logs.jsonl
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.agent import build_prompt, call_model
from src.gate import apply_gate
from src.parser import parse_model_output
from src.registry import load_registry
from src.run_log import RunLogger, generate_run_id

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and lightly validate experiment YAML config.

    Required keys: experiment.name, experiment.seed, model.name,
    model.endpoint, registry.variant, dataset.path.
    """
    text = config_path.read_text(encoding="utf-8")
    cfg: dict[str, Any] = yaml.safe_load(text)

    required_paths = [
        ("experiment", "name"),
        ("experiment", "seed"),
        ("model", "name"),
        ("model", "endpoint"),
        ("registry", "variant"),
        ("dataset", "path"),
    ]
    for section, key in required_paths:
        if section not in cfg or key not in cfg[section]:
            raise ValueError(f"Config missing required key: {section}.{key}")

    cfg.setdefault("experiment", {}).setdefault("repeats", 3)
    cfg.setdefault("gate", {}).setdefault("enabled", False)
    cfg.setdefault("output", {}).setdefault("base_dir", "runs")
    cfg.setdefault("output", {}).setdefault("run_id", None)
    cfg.setdefault("output", {}).setdefault("logs_file", "logs.jsonl")

    return cfg


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load JSONL final dataset; skip blank lines and comments."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(json.loads(s))
    return rows


def _prompt_hash(messages: list[dict[str, Any]]) -> str:
    """SHA-256 of the serialized messages, truncated to 16 hex chars."""
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _label_source(record: dict[str, Any]) -> str:
    """Derive label_source tag for the log from the final record fields."""
    if record.get("source_labeled_by") == "llm_draft":
        return "llm_draft"
    return "human_written"


def _results_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Compare two parsed dicts for consistency (kind + calls)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get("kind") == b.get("kind") and json.dumps(
        a.get("calls", []), sort_keys=True
    ) == json.dumps(b.get("calls", []), sort_keys=True)


def _check_consistency(repeat_results: list[dict[str, Any]]) -> bool:
    """Return True iff all repeats produced identical parsed outputs."""
    parsed_list = [r["parsed"] for r in repeat_results]
    if not parsed_list:
        return True
    first = parsed_list[0]
    return all(_results_equal(first, p) for p in parsed_list[1:])


def run_experiment(
    config_path: Path,
    *,
    schemas_dir: Path | None = None,
) -> RunLogger:
    """Execute the full evaluation loop and return the bound RunLogger.

    Args:
        config_path: Path to the experiment YAML config file.
        schemas_dir: Override for the schemas directory (used in tests).

    The runner never retries a failed model call — failures are logged as-is
    (parse_success=False) and counted as data per the project invariant.
    """
    cfg = load_config(config_path)

    exp_name: str = cfg["experiment"]["name"]
    repeats: int = int(cfg["experiment"].get("repeats", 3))
    gate_enabled: bool = bool(cfg.get("gate", {}).get("enabled", False))

    variant = cfg["registry"]["variant"]
    kwargs: dict[str, Any] = {}
    if schemas_dir is not None:
        kwargs["schemas_dir"] = schemas_dir
    registry = load_registry(variant, **kwargs)

    dataset_raw = Path(cfg["dataset"]["path"])
    dataset_path = dataset_raw if dataset_raw.is_absolute() else ROOT / dataset_raw
    records = _load_dataset(dataset_path)

    run_id: str | None = cfg["output"].get("run_id") or None
    if run_id is None:
        run_id = generate_run_id(exp_name)

    base_dir_raw = cfg["output"].get("base_dir", "runs")
    base_dir_path = Path(base_dir_raw)
    base_dir = base_dir_path if base_dir_path.is_absolute() else ROOT / base_dir_raw
    logs_file: str = cfg["output"].get("logs_file", "logs.jsonl")

    logger = RunLogger(run_id, base_dir=base_dir, logs_filename=logs_file)

    # Write a run manifest as the first log entry
    logger.log(
        {
            "entry_type": "run_manifest",
            "run_id": run_id,
            "experiment": exp_name,
            "model": cfg["model"]["name"],
            "registry_variant": variant,
            "seed": cfg["experiment"]["seed"],
            "repeats": repeats,
            "gate_enabled": gate_enabled,
            "dataset_path": str(dataset_path),
            "record_count": len(records),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    for record in records:
        record_id: str = record.get("id", "")
        utterance: str = record.get("utterance", "")
        vehicle_state: dict[str, Any] = record.get("vehicle_state", {})
        utterance_type: str = record.get("utterance_type", "")
        label_src = _label_source(record)

        messages = build_prompt(registry, utterance, vehicle_state)
        p_hash = _prompt_hash(messages)

        repeat_results: list[dict[str, Any]] = []

        for repeat_idx in range(repeats):
            entry: dict[str, Any] = {
                "entry_type": "inference",
                "record_id": record_id,
                "utterance_type": utterance_type,
                "label_source": label_src,
                "utterance": utterance,
                "vehicle_state": vehicle_state,
                "repeat_idx": repeat_idx,
                "prompt_hash": p_hash,
                "registry_variant": variant,
                "model": cfg["model"]["name"],
                "seed": cfg["experiment"]["seed"],
                "gate_enabled": gate_enabled,
                "raw_output": None,
                "parse_success": False,
                "parse_error": None,
                "parsed": None,
                "ttft_ms": None,
                "total_ms": None,
                "gate_applied": False,
                "gate_events": [],
                "consistency": None,
            }

            try:
                call_result = call_model(messages, cfg)
                entry["raw_output"] = call_result["raw_output"]
                entry["ttft_ms"] = call_result["ttft_ms"]
                entry["total_ms"] = call_result["total_ms"]

                parse_result = parse_model_output(call_result["raw_output"])
                entry["parse_success"] = parse_result["parse_success"]
                entry["parse_error"] = parse_result["parse_error"]
                entry["parsed"] = parse_result["parsed"]

                if gate_enabled and parse_result["parse_success"] and parse_result["parsed"]:
                    p = parse_result["parsed"]
                    new_kind, gate_events = apply_gate(
                        p["kind"], p["calls"], vehicle_state, registry
                    )
                    if gate_events:
                        entry["gate_applied"] = True
                        entry["gate_events"] = gate_events
                        entry["parsed"] = {**p, "kind": new_kind}

            except Exception as exc:  # network errors, timeouts, etc.
                entry["parse_error"] = f"{type(exc).__name__}: {exc}"

            repeat_results.append(entry)

        # Compute consistency across repeats for this record
        consistent = _check_consistency(repeat_results)
        for r in repeat_results:
            r["consistency"] = consistent

        for r in repeat_results:
            logger.log(r)

    # Write a run summary
    logger.log(
        {
            "entry_type": "run_summary",
            "run_id": run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "records_processed": len(records),
        }
    )

    return logger
