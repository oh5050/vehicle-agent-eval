"""Normalizer: synonym mapping, numeric coercion, default fill, equivalence rules.

Pipeline position: parser → registry.validate_call → **normalizer** → rule scorer.

Basic normalization (applied in both strict and normalized scoring):
- synonym mapping from data/synonyms.yaml (param → alias → canonical)
- numeric coercion: string numbers → int/float; range violations are RECORDED,
  never clamped
- default fill: schema defaults are filled symmetrically on gold AND pred
  (E-2: climate_set zone omitted == zone=driver falls out of this)

Equivalence rules (normalized scoring only):
- E-1: window_control called once per window (driver, passenger, rear_left,
  rear_right) with otherwise-identical args == a single window=all call.
  Both forms canonicalize to the window=all form. seat_control has no "all"
  concept, so the rule does not apply to it.
- E-3: every applied rule is recorded in the transformation history.
- E-4: callers score both before (strict) and after (normalized) equivalence.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"

SYNONYM_PARAM_KEYS = ("window", "seat", "zone", "mode", "item")

_SYNONYMS_SKELETON = """\
# Normalization synonyms for rule-based scoring (Stage 4 normalizer).
#
# Structure: <param_name> -> { <alias>: <canonical enum value> }
# 사람이 관리하는 파일 — 코드는 읽기만 한다.

window: {}
seat: {}
zone: {}
mode: {}
item: {}
"""

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

_WINDOW_ALL_PARTS = frozenset({"driver", "passenger", "rear_left", "rear_right"})


def ensure_synonyms_file(path: Path | str = DEFAULT_SYNONYMS_PATH) -> Path:
    """Create a skeleton synonyms file if missing; return the path."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_SYNONYMS_SKELETON, encoding="utf-8")
    return p


def load_synonyms(path: Path | str = DEFAULT_SYNONYMS_PATH) -> dict[str, dict[str, str]]:
    """Load param → alias → canonical maps. Missing file → skeleton + empty maps."""
    p = ensure_synonyms_file(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    result: dict[str, dict[str, str]] = {}
    for param, mapping in raw.items():
        if isinstance(mapping, dict):
            result[param] = {str(k): str(v) for k, v in mapping.items()}
        else:
            result[param] = {}
    return result


@dataclass
class NormalizationResult:
    calls: list[dict[str, Any]]
    history: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def _coerce_numeric(value: Any, expected_type: str | None) -> tuple[Any, bool]:
    """Convert string numbers to int/float per schema type. Returns (value, changed)."""
    if not isinstance(value, str) or expected_type not in ("integer", "number"):
        return value, False
    if not _NUMBER_RE.match(value.strip()):
        return value, False
    num = float(value.strip())
    if expected_type == "integer" and num.is_integer():
        return int(num), True
    return (int(num) if num.is_integer() and expected_type == "integer" else num), True


def _normalize_single_call(
    call: dict[str, Any],
    registry: Registry,
    synonyms: dict[str, dict[str, str]],
    history: list[dict[str, Any]],
    violations: list[str],
) -> dict[str, Any]:
    tool_name = call.get("tool", "")
    args = dict(call.get("args") or {})
    tool = registry.get_tool(tool_name)

    for param, value in list(args.items()):
        # 1. synonym mapping
        param_map = synonyms.get(param) or {}
        if isinstance(value, str) and value in param_map:
            canonical = param_map[value]
            args[param] = canonical
            history.append(
                {"rule": "synonym", "tool": tool_name, "param": param, "from": value, "to": canonical}
            )
            value = canonical

        if tool is None:
            continue
        spec = tool.parameters.get(param)
        if spec is None:
            continue

        # 2. numeric coercion (no clamping)
        coerced, changed = _coerce_numeric(value, spec.type)
        if changed:
            args[param] = coerced
            history.append(
                {"rule": "numeric_coerce", "tool": tool_name, "param": param, "from": value, "to": coerced}
            )
            value = coerced

        # 3. range violation — record only
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.minimum is not None and value < spec.minimum:
                violations.append(
                    f"{tool_name}.{param}={value} < minimum {spec.minimum}"
                )
            if spec.maximum is not None and value > spec.maximum:
                violations.append(
                    f"{tool_name}.{param}={value} > maximum {spec.maximum}"
                )

    # 4. default fill (symmetric — caller applies to both gold and pred)
    if tool is not None:
        for param, spec in tool.parameters.items():
            if spec.default is not None and param not in args:
                args[param] = spec.default
                rule = "E-2" if tool_name == "climate_set" and param == "zone" else "default_fill"
                history.append(
                    {"rule": rule, "tool": tool_name, "param": param, "to": spec.default}
                )

    return {"tool": tool_name, "args": args}


def _args_without(args: dict[str, Any], key: str) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, _freeze(v)) for k, v in args.items() if k != key))


def _freeze(value: Any) -> Any:
    """Convert a value to a sortable, hashable representation for signatures."""
    if value is None:
        return ("__none__",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("num", float(value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((k, _freeze(v)) for k, v in value.items())))
    return ("str", str(value))


def _apply_e1_window_all(
    calls: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse per-window window_control calls covering all 4 windows into window=all."""
    groups: dict[tuple, list[int]] = {}
    for i, call in enumerate(calls):
        if call.get("tool") != "window_control":
            continue
        args = call.get("args") or {}
        window = args.get("window")
        if window not in _WINDOW_ALL_PARTS:
            continue
        key = _args_without(args, "window")
        groups.setdefault(key, []).append(i)

    to_remove: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}

    for key, indices in groups.items():
        windows = {calls[i]["args"].get("window") for i in indices}
        if windows == set(_WINDOW_ALL_PARTS):
            first = min(indices)
            merged_args = dict(calls[first]["args"])
            merged_args["window"] = "all"
            replacements[first] = {"tool": "window_control", "args": merged_args}
            to_remove.update(i for i in indices if i != first)
            history.append(
                {
                    "rule": "E-1",
                    "tool": "window_control",
                    "detail": "4 per-window calls collapsed to window=all",
                    "collapsed_indices": sorted(indices),
                }
            )

    if not replacements:
        return calls

    result: list[dict[str, Any]] = []
    for i, call in enumerate(calls):
        if i in to_remove:
            continue
        result.append(replacements.get(i, call))
    return result


def normalize_calls(
    calls: list[dict[str, Any]],
    *,
    registry: Registry,
    synonyms: dict[str, dict[str, str]] | None = None,
    equivalence: bool = True,
) -> NormalizationResult:
    """Normalize a call list. equivalence=False → strict (basic normalization only)."""
    syn = synonyms or {}
    history: list[dict[str, Any]] = []
    violations: list[str] = []

    normalized = [
        _normalize_single_call(copy.deepcopy(c), registry, syn, history, violations)
        for c in (calls or [])
    ]

    if equivalence:
        normalized = _apply_e1_window_all(normalized, history)

    return NormalizationResult(calls=normalized, history=history, violations=violations)


def calls_signature(calls: list[dict[str, Any]], *, ordered: bool) -> tuple:
    """Comparable signature of a call list (order-sensitive or not)."""
    sigs = [
        (c.get("tool", ""), tuple(sorted((k, _freeze(v)) for k, v in (c.get("args") or {}).items())))
        for c in calls
    ]
    return tuple(sigs) if ordered else tuple(sorted(sigs))
