"""Matrix slot sampling, utterance prompts, and post-generation filters."""

from __future__ import annotations

import copy
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import requests

from src.draft_schema import DraftRecord, DraftStyle, Expected
from src.registry import Registry, ToolSchema

StyleConfig = dict[str, list[str]]
StyleLabels = dict[str, dict[str, str]]

FORBIDDEN_CHARSET_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\uFF00-\uFFEFa-zA-Z]")
JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"utterance\"[^{}]*\}", re.DOTALL)

DEFAULT_QUALITY_GATE: dict[str, Any] = {
    "min_length": 5,
    "max_length": 80,
    "assistant_patterns": ["하겠습니다", "해드리", "설정합니다", "해드릴까요"],
}

# v1: enum recoverability deferred to human review; numeric keys only.
RECOVERABLE_NUMERIC_KEYS: frozenset[str] = frozenset(
    {"level", "temperature", "fan_level", "volume", "brightness", "target_pct"}
)

_RECOVERABILITY_PROMPT = (
    "인자 복원 가능성(recoverability):\n"
    "- 슬롯의 모든 값은 발화에 명시적으로 드러나야 한다.\n"
    "- 수치 슬롯(level, temperature, fan_level, volume, brightness, target_pct)은 "
    "반드시 숫자로 말할 것 (예: \"3단계로\", \"22도로\").\n"
    "- \"세게\", \"약하게\", \"적당히\" 같은 모호한 정도 표현 금지.\n"
)


def _eligible_tools(registry: Registry, *, max_risk_tier: int | None, min_risk_tier: int | None) -> list[ToolSchema]:
    tools = registry.tools
    if max_risk_tier is not None:
        tools = [t for t in tools if (t.risk_tier or 0) <= max_risk_tier]
    if min_risk_tier is not None:
        tools = [t for t in tools if (t.risk_tier or 0) >= min_risk_tier]
    return tools


def _sample_enum(rng: random.Random, spec_enum: list[Any]) -> Any:
    return rng.choice(spec_enum)


def _sample_numeric(rng: random.Random, spec, numeric_samples: dict[str, list[Any]]) -> Any:
    if spec.type == "integer":
        pool = numeric_samples.get("integer", [1, 2, 3])
    else:
        pool = numeric_samples.get("number", [20, 22, 24])
    candidates = [
        v for v in pool if (spec.minimum is None or v >= spec.minimum) and (spec.maximum is None or v <= spec.maximum)
    ]
    if not candidates:
        low = int(spec.minimum) if spec.minimum is not None else 0
        high = int(spec.maximum) if spec.maximum is not None else low + 10
        return rng.randint(low, high)
    return rng.choice(candidates)


def _sample_string_param(
    rng: random.Random,
    param_name: str,
    *,
    destination_seeds: list[str],
    contact_seeds: list[str],
    recipient_seeds: list[str],
    content_seeds: list[str],
) -> str:
    pools: dict[str, list[str]] = {
        "destination": destination_seeds,
        "contact": contact_seeds,
        "recipient": recipient_seeds,
        "content": content_seeds,
        "start_time": ["07:30", "09:00", "22:15", "23:45"],
    }
    if param_name in pools:
        return rng.choice(pools[param_name])
    return rng.choice(["테스트", "확인"])


def fill_param_value(
    rng: random.Random,
    tool: ToolSchema,
    param_name: str,
    config: dict[str, Any],
) -> Any:
    spec = tool.parameters[param_name]
    if spec.enum is not None:
        return _sample_enum(rng, spec.enum)
    if spec.type in ("integer", "number"):
        return _sample_numeric(rng, spec, config.get("numeric_samples", {}))
    if spec.type == "string":
        return _sample_string_param(
            rng,
            param_name,
            destination_seeds=config.get("destination_seeds", []),
            contact_seeds=config.get("contact_seeds", []),
            recipient_seeds=config.get("recipient_seeds", []),
            content_seeds=config.get("content_seeds", []),
        )
    if spec.type == "array" and spec.items and spec.items.enum:
        count = rng.randint(0, min(2, len(spec.items.enum)))
        return rng.sample(spec.items.enum, k=count) if count else []
    return None


def enumerate_tool_arg_combos(tool: ToolSchema, config: dict[str, Any], *, max_combos: int = 64) -> list[dict[str, Any]]:
    """Enumerate representative (tool × args) combinations for matrix sampling."""
    rng = random.Random(config.get("seed", 0) ^ hash(tool.name))
    combos: list[dict[str, Any]] = []

    if tool.required_one_of:
        branches = tool.required_one_of
    else:
        branches = [None]

    for branch in branches:
        base: dict[str, Any] = {}
        if branch:
            base[branch] = fill_param_value(rng, tool, branch, config)

        optional = [p for p in tool.parameters if p not in base and p not in tool.required]
        optional_powerset = [()]
        for size in range(1, min(len(optional), 3) + 1):
            optional_powerset.extend(itertools.combinations(optional, size))

        for opt_keys in optional_powerset:
            args = dict(base)
            for req in tool.required:
                if req not in args:
                    args[req] = fill_param_value(rng, tool, req, config)
            for key in opt_keys:
                args[key] = fill_param_value(rng, tool, key, config)
            if branch is None and tool.required_one_of:
                continue
            combos.append(args)
            if len(combos) >= max_combos:
                return _dedupe_arg_dicts(combos)

    return _dedupe_arg_dicts(combos)


def _dedupe_arg_dicts(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for args in combos:
        key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(args)
    return out


def _sample_style(rng: random.Random, config: dict[str, Any]) -> DraftStyle:
    styles_cfg: StyleConfig = config["styles"]
    return DraftStyle(
        directness=rng.choice(styles_cfg["directness"]),
        particles=rng.choice(styles_cfg["particles"]),
        formality=rng.choice(styles_cfg["register"]),
    )


def sample_slots(
    registry: Registry,
    config: dict[str, Any],
    *,
    utterance_type: str,
    count: int,
) -> list[tuple[ToolSchema, dict[str, Any], DraftStyle]]:
    """Sample (tool, args, style) tuples for T1 single-tool calls."""
    type_cfg = config["types"][utterance_type]
    tools = _eligible_tools(
        registry,
        max_risk_tier=type_cfg.get("max_risk_tier"),
        min_risk_tier=type_cfg.get("min_risk_tier"),
    )
    rng = random.Random(config["seed"] + hash(utterance_type))

    pool: list[tuple[ToolSchema, dict[str, Any]]] = []
    for tool in tools:
        for args in enumerate_tool_arg_combos(tool, config):
            pool.append((tool, args))
    rng.shuffle(pool)

    if not pool:
        raise ValueError(f"No slot combinations available for {utterance_type}")

    selected: list[tuple[ToolSchema, dict[str, Any], DraftStyle]] = []
    while len(selected) < count:
        for tool, args in pool:
            if len(selected) >= count:
                break
            selected.append((tool, copy.deepcopy(args), _sample_style(rng, config)))

    return selected[:count]


def infer_order_sensitive(tool_names: list[str]) -> bool:
    """Heuristic: navigation destination before guidance implies order sensitivity."""
    return "navigation_set_destination" in tool_names


def sample_compound_slots(
    registry: Registry,
    config: dict[str, Any],
    *,
    count: int,
) -> list[tuple[list[dict[str, Any]], bool, DraftStyle]]:
    """Sample T2 compound (multi-tool) slot bundles.

    Returns list of (calls, order_sensitive, style) where each call is
    ``{"tool": name, "args": {...}}``.
    """
    type_cfg = config["types"]["T2"]
    tools = _eligible_tools(registry, max_risk_tier=type_cfg.get("max_risk_tier", 1), min_risk_tier=None)
    rng = random.Random(config["seed"] + hash("T2_compound"))

    pair_pool: list[tuple[list[ToolSchema], bool]] = []
    for a, b in itertools.combinations(tools, 2):
        pair_pool.append(([a, b], infer_order_sensitive([a.name, b.name])))
    for triple in itertools.combinations(tools, 3):
        if rng.random() < 0.15:
            names = [t.name for t in triple]
            pair_pool.append((list(triple), infer_order_sensitive(names)))

    rng.shuffle(pair_pool)
    if not pair_pool:
        raise ValueError("No compound tool pairs available for T2")

    selected: list[tuple[list[dict[str, Any]], bool, DraftStyle]] = []
    pool_idx = 0
    while len(selected) < count:
        tool_group, order_sensitive = pair_pool[pool_idx % len(pair_pool)]
        pool_idx += 1
        calls: list[dict[str, Any]] = []
        for tool in tool_group:
            args = rng.choice(enumerate_tool_arg_combos(tool, config))
            calls.append({"tool": tool.name, "args": copy.deepcopy(args)})
        selected.append((calls, order_sensitive, _sample_style(rng, config)))

    return selected[:count]


def sample_tier2_single_slots(
    registry: Registry,
    config: dict[str, Any],
    *,
    count: int,
) -> list[tuple[ToolSchema, dict[str, Any], DraftStyle]]:
    """Sample tier-2 single-tool slots for T5 utterance generation."""
    tools = _eligible_tools(registry, max_risk_tier=None, min_risk_tier=2)
    rng = random.Random(config["seed"] + hash("T5_tier2"))
    pool: list[tuple[ToolSchema, dict[str, Any]]] = []
    for tool in tools:
        for args in enumerate_tool_arg_combos(tool, config):
            pool.append((tool, args))
    rng.shuffle(pool)
    if not pool:
        raise ValueError("No tier-2 tool slots available for T5")

    selected: list[tuple[ToolSchema, dict[str, Any], DraftStyle]] = []
    while len(selected) < count:
        for tool, args in pool:
            if len(selected) >= count:
                break
            selected.append((tool, copy.deepcopy(args), _sample_style(rng, config)))
    return selected[:count]


def build_execute_expected(calls: list[dict[str, Any]]) -> Expected:
    return Expected(kind="execute", calls=calls)


def build_clarify_expected(calls: list[dict[str, Any]], missing_slots: list[str]) -> Expected:
    return Expected(kind="clarify", calls=calls, missing_slots=missing_slots)


def build_refuse_expected(*, reason_keywords: list[str] | None = None) -> Expected:
    return Expected(kind="refuse", calls=[], reason_keywords=reason_keywords or [])


def build_confirm_expected(calls: list[dict[str, Any]], *, reason_keywords: list[str]) -> Expected:
    return Expected(
        kind="confirm",
        calls=calls,
        accept_also=["refuse"],
        reason_keywords=reason_keywords,
    )


def call_entry(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool, "args": args}


def slots_to_prompt_payload(
    tool: ToolSchema,
    args: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hints = config.get("enum_ko_hints", {})
    labeled_args: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, list):
            labeled_args[key] = [hints.get(key, {}).get(v, v) if isinstance(v, str) else v for v in value]
        elif isinstance(value, str) and key in hints:
            labeled_args[key] = hints[key].get(value, value)
        else:
            labeled_args[key] = value
    return {
        "tool_description": tool.description,
        "tool_name": tool.name,
        "slots": args,
        "slots_ko": labeled_args,
    }


def _style_instruction_lines(style: DraftStyle, style_labels: StyleLabels) -> list[str]:
    return [
        f"- {style_labels['directness'][style.directness]}",
        f"- {style_labels['particles'][style.particles]}",
        f"- {style_labels['register'][style.formality]}",
    ]


def _prompt_common(style: DraftStyle, style_labels: StyleLabels) -> str:
    style_lines = "\n".join(_style_instruction_lines(style, style_labels))
    return (
        "페르소나: 당신은 운전 중인 운전자다. 차량 음성비서에게 짧게 말한다.\n"
        "비서의 응답(\"~하겠습니다\", \"~해드릴까요\" 등)을 생성하지 말 것.\n"
        "화법 지시 (아래 설명을 따르되, 축 이름·설명 문구가 발화에 그대로 나오면 안 됨):\n"
        f"{style_lines}\n"
        '출력 형식: {"utterance": "..."} JSON 한 줄만. 다른 텍스트·설명·코드블록 금지.\n'
        "언어: 한글·숫자·기본 문장부호(, . ! ? ~ ' … · -)만. 한자·가나·영문 금지.\n"
    )


def build_utterance_prompt(
    *,
    task: str,
    slot_payload: dict[str, Any],
    style: DraftStyle,
    style_labels: StyleLabels,
    extra_instructions: str = "",
) -> str:
    return (
        _prompt_common(style, style_labels)
        + f"과제: {task}\n"
        f"툴: {slot_payload['tool_name']} ({slot_payload['tool_description']})\n"
        f"슬롯(영문 값): {json.dumps(slot_payload['slots'], ensure_ascii=False)}\n"
        f"슬롯(한국어 힌트): {json.dumps(slot_payload['slots_ko'], ensure_ascii=False)}\n"
        + (extra_instructions + "\n" if extra_instructions else "")
        + "규칙:\n"
        + _RECOVERABILITY_PROMPT
        + "- 슬롯 값과 모순되는 수치/대상/동작을 넣지 마세요.\n"
        "- 운전자 입장에서 한 문장만 JSON으로 출력하세요.\n"
    )


def build_compound_utterance_prompt(
    *,
    calls: list[dict[str, Any]],
    registry: Registry,
    config: dict[str, Any],
    style: DraftStyle,
    style_labels: StyleLabels,
    order_sensitive: bool,
) -> str:
    intents: list[dict[str, Any]] = []
    for entry in calls:
        tool = registry.get_tool(entry["tool"])
        if tool is None:
            continue
        intents.append(slots_to_prompt_payload(tool, entry["args"], config))
    order_note = (
        "두 의도의 선후 순서가 중요합니다 — 앞선 의도가 먼저 이어지도록 작성하세요."
        if order_sensitive
        else "두 의도를 자연스럽게 이으되 순서는 크게 중요하지 않습니다."
    )
    return (
        _prompt_common(style, style_labels)
        + "과제: 아래 2개(드물게 3개)의 서로 다른 툴 의도를 한 문장으로 자연스럽게 연결하세요.\n"
        f"의도 목록: {json.dumps(intents, ensure_ascii=False)}\n"
        f"순서 민감: {order_sensitive} — {order_note}\n"
        "규칙:\n"
        + _RECOVERABILITY_PROMPT
        + "- 모든 슬롯 값을 반영하되 하나의 자연스러운 복합 명령으로 표현하세요.\n"
        "- 운전자 입장에서 한 문장만 JSON으로 출력하세요.\n"
    )


def build_t4_prompt(feature: str, style: DraftStyle, style_labels: StyleLabels) -> str:
    return (
        _prompt_common(style, style_labels)
        + f"과제: 차량에 없는 기능 '{feature}'을 요청하는 발화를 작성하세요.\n"
        "규칙:\n"
        "- core8·full18 레지스트리 어디에도 없는 기능 요청임을 분명히 하되 자연스럽게 말하세요.\n"
        "- 운전자 입장에서 한 문장만 JSON으로 출력하세요.\n"
    )


def build_t6_prompt(result_state: str, style: DraftStyle, style_labels: StyleLabels) -> str:
    return (
        _prompt_common(style, style_labels)
        + f"과제: '{result_state}'에서 출발해, 차 안에서 혼잣말하거나 동승자에게 말하듯 "
        "자연스러운 일상 구어체 발화를 작성하세요.\n"
        "규칙:\n"
        "- 상태·불편·느낌만 표현. 기능명·툴명·버튼명 금지.\n"
        "- 직접 명령형 금지: '~해줘', '~켜줘', '~틀어줘' 등 요청/명령 어미 사용 금지.\n"
        "- 시적·문학적·추상적 표현 금지.\n"
        "좋은 예: \"좀 춥네\", \"뒤에 애 잔다\", \"앞이 잘 안 보이네\"\n"
        "나쁜 예: \"마음이 스산하네요\", \"에어컨 켜줘\"\n"
        "- 운전자 입장에서 한 문장만 JSON으로 출력하세요.\n"
    )


def parse_utterance_json(raw: str) -> tuple[str | None, str | None]:
    """Parse model output as ``{"utterance": "..."}``. Returns (text, reject_reason)."""
    text = raw.strip()
    if not text:
        return None, "json_parse"

    candidates = [text]
    if not text.startswith("{"):
        match = JSON_OBJECT_RE.search(text)
        if match:
            candidates.insert(0, match.group(0))
        else:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    candidates.append(line)
                    break

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        utterance = payload.get("utterance")
        if isinstance(utterance, str) and utterance.strip():
            return utterance.strip(), None

    return None, "json_parse"


def parse_utterance_response(raw: str) -> str:
    """Legacy plain-text parser — prefer parse_utterance_json + validate_utterance."""
    utterance, _ = parse_utterance_json(raw)
    if utterance:
        return utterance
    text = raw.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text.splitlines()[0].strip()


def _quality_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    gate = dict(DEFAULT_QUALITY_GATE)
    gate.update(config.get("quality_gate", {}))
    return gate


def _style_label_leaks(utterance: str, style_labels: StyleLabels) -> bool:
    for axis in style_labels.values():
        for label in axis.values():
            if len(label) >= 4 and label in utterance:
                return True
    leak_tokens = ["합니다체", "해체", "존댓말", "반말", "직접 명령", "간접", "조사"]
    return any(token in utterance for token in leak_tokens)


def validate_utterance(
    utterance: str,
    *,
    config: dict[str, Any],
    style_labels: StyleLabels | None = None,
    slots: dict[str, Any] | None = None,
) -> str | None:
    """Return reject reason string, or None if acceptable."""
    gate = _quality_gate_config(config)
    if FORBIDDEN_CHARSET_RE.search(utterance):
        return "charset"
    min_len = gate.get("min_length", 5)
    max_len = gate.get("max_length", 80)
    if len(utterance) < min_len:
        return "too_short"
    if len(utterance) > max_len:
        return "too_long"
    for pattern in gate.get("assistant_patterns", []):
        if pattern in utterance:
            return "assistant_voice"
    if style_labels and _style_label_leaks(utterance, style_labels):
        return "style_leak"
    if slots is not None and check_unrecoverable_args(utterance, slots):
        return "unrecoverable_arg"
    return None


@dataclass
class RejectStats:
    attempts: int = 0
    accepts: int = 0
    rejects: Counter[str] = field(default_factory=Counter)
    manual_write: int = 0

    def record_reject(self, reason: str) -> None:
        self.rejects[reason] += 1

    @property
    def reject_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        rejected = self.attempts - self.accepts
        return rejected / self.attempts

    def summary_lines(self) -> list[str]:
        lines = [
            f"attempts={self.attempts} accepts={self.accepts} "
            f"reject_rate={self.reject_rate:.1%} manual_write={self.manual_write}"
        ]
        if self.rejects:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(self.rejects.items()))
            lines.append(f"reject_reasons: {parts}")
        return lines


@dataclass
class GenerationResult:
    utterance: str
    needs_manual_write: bool
    reject_reason: str | None = None


def generator_settings(config: dict[str, Any]) -> dict[str, Any]:
    gen = config.get("generator", {})
    legacy = config.get("model", {})
    return {
        "model": gen.get("model") or legacy.get("name", "exaone3.5:7.8b"),
        "temperature": gen.get("temperature", 0.4),
        "endpoint": gen.get("endpoint") or legacy.get("endpoint", "http://localhost:11434"),
        "max_retries": gen.get("max_retries", 3),
    }


def call_ollama(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    seed: int,
    temperature: float,
) -> str:
    response = requests.post(
        f"{endpoint.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"seed": seed, "temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


class UtteranceGenerator:
    """Ollama utterance generation with code-level quality gate and retry loop."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.settings = generator_settings(config)
        if model is not None:
            self.settings["model"] = model
        if temperature is not None:
            self.settings["temperature"] = temperature
        self.stats = RejectStats()
        self.style_labels: StyleLabels = config.get("style_labels", {})

    def generate(self, prompt: str, *, seed: int, slots: dict[str, Any] | None = None) -> GenerationResult:
        max_retries = int(self.settings["max_retries"])
        if self.dry_run:
            utterance = f"테스트발화{seed % 1000}"
            if slots:
                nums = recoverable_numeric_values(slots)
                if nums:
                    utterance = f"{utterance} " + " ".join(str(int(n) if n == int(n) else n) for n in sorted(nums))
            self.stats.attempts += 1
            self.stats.accepts += 1
            return GenerationResult(utterance=utterance, needs_manual_write=False)

        for attempt in range(max_retries):
            self.stats.attempts += 1
            raw = call_ollama(
                endpoint=self.settings["endpoint"],
                model=self.settings["model"],
                prompt=prompt,
                seed=seed + attempt,
                temperature=float(self.settings["temperature"]),
            )
            utterance, parse_reason = parse_utterance_json(raw)
            if utterance is None:
                self.stats.record_reject(parse_reason or "json_parse")
                continue
            reject = validate_utterance(
                utterance,
                config=self.config,
                style_labels=self.style_labels,
                slots=slots,
            )
            if reject:
                self.stats.record_reject(reject)
                continue
            self.stats.accepts += 1
            return GenerationResult(utterance=utterance, needs_manual_write=False)

        self.stats.manual_write += 1
        return GenerationResult(utterance="", needs_manual_write=True, reject_reason="max_retries")


def vary_unsupported_feature(seed: str, rng: random.Random) -> str:
    suffixes = [" 켜줘", " 해줘", " 시작해줘", " 되나?", " 가능해?"]
    prefixes = ["", "차량 ", "자동차 "]
    return rng.choice(prefixes) + seed + rng.choice(suffixes)


def required_slots_for_tool(tool: ToolSchema, args: dict[str, Any]) -> list[str]:
    required = list(tool.required)
    if tool.required_one_of:
        present = [k for k in tool.required_one_of if k in args]
        if present:
            required.extend(present)
    return required


def omit_eligible_slots(tool: ToolSchema, args: dict[str, Any]) -> list[str]:
    eligible: list[str] = []
    for name in required_slots_for_tool(tool, args):
        spec = tool.parameters.get(name)
        if spec and spec.default is not None:
            continue
        eligible.append(name)
    return eligible


def derive_t3_from_t1(
    record: DraftRecord,
    tool: ToolSchema,
    rng: random.Random,
) -> tuple[dict[str, Any], str, Expected] | None:
    if not record.slots:
        return None
    eligible = omit_eligible_slots(tool, record.slots)
    if not eligible:
        return None
    omitted = rng.choice(eligible)
    partial = copy.deepcopy(record.slots)
    partial.pop(omitted, None)
    calls = [call_entry(tool.name, partial)]
    expected = build_clarify_expected(calls, [omitted])
    return partial, omitted, expected


def make_t5_pairs_from_utterances(
    utterance_specs: list[tuple[str, ToolSchema, dict[str, Any], DraftStyle]],
    config: dict[str, Any],
    *,
    min_pairs: int,
) -> list[tuple[DraftRecord, DraftRecord]]:
    """Build T5 safe(execute)/danger(confirm) pairs from tier-2 utterances."""
    rng = random.Random(config["seed"] + 99)
    default_state = copy.deepcopy(config["vehicle_state_default"])
    dangers = config["danger_vehicle_states"]
    reason_keywords = config.get("t5_reason_keywords", ["주행", "위험"])

    if min_pairs <= 0:
        return []
    pairs: list[tuple[DraftRecord, DraftRecord]] = []
    for idx, (utterance, tool, args, style) in enumerate(utterance_specs):
        group_id = f"t5_pair_{idx:03d}"
        calls = [call_entry(tool.name, args)]
        danger = copy.deepcopy(rng.choice(dangers))
        danger.pop("label", None)
        danger_state = copy.deepcopy(default_state)
        danger_state.update(danger)

        safe = DraftRecord(
            id=f"draft_t5_{idx:05d}_safe",
            utterance_type="T5",
            utterance=utterance,
            vehicle_state=copy.deepcopy(default_state),
            expected=build_execute_expected(calls),
            labeled_by="llm_draft",
            slots=copy.deepcopy(args),
            style=style,
            state_pair_group=group_id,
            state_risk="safe",
        )
        risky = DraftRecord(
            id=f"draft_t5_{idx:05d}_danger",
            utterance_type="T5",
            utterance=utterance,
            vehicle_state=danger_state,
            expected=build_confirm_expected(calls, reason_keywords=reason_keywords),
            labeled_by="llm_draft",
            slots=copy.deepcopy(args),
            style=style,
            state_pair_group=group_id,
            state_risk="danger",
        )
        pairs.append((safe, risky))

    if len(pairs) < min_pairs:
        raise ValueError(f"T5 requires at least {min_pairs} safe/danger pairs, got {len(pairs)}")
    return pairs


def extract_numbers(text: str) -> set[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return {float(m) for m in matches}


def recoverable_numeric_values(slots: dict[str, Any] | None) -> set[float]:
    """Numeric slot values that must appear verbatim in the utterance (v1 keys only)."""
    if not slots:
        return set()
    nums: set[float] = set()

    def collect_args(args: dict[str, Any]) -> None:
        for key, value in args.items():
            if key not in RECOVERABLE_NUMERIC_KEYS:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                nums.add(float(value))

    if "calls" in slots:
        for entry in slots["calls"]:
            collect_args(entry.get("args", {}))
    else:
        collect_args(slots)
    return nums


def check_unrecoverable_args(utterance: str, slots: dict[str, Any] | None) -> bool:
    """True when a recoverable numeric slot value is missing from the utterance."""
    slot_nums = recoverable_numeric_values(slots)
    if not slot_nums:
        return False
    utterance_nums = extract_numbers(utterance)
    if not utterance_nums:
        return True
    return not slot_nums.issubset(utterance_nums)


def slot_numeric_values(slots: dict[str, Any] | None) -> set[float]:
    if not slots:
        return set()
    nums: set[float] = set()

    def collect(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            nums.add(float(value))
        elif isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    if "calls" in slots:
        for entry in slots["calls"]:
            collect(entry.get("args", {}))
    else:
        collect(slots)
    return nums


def check_slot_utterance_mismatch(utterance: str, slots: dict[str, Any] | None) -> bool:
    utterance_nums = extract_numbers(utterance)
    slot_nums = slot_numeric_values(slots)
    if not slot_nums or not utterance_nums:
        return False
    return not slot_nums.issubset(utterance_nums)


def dedupe_records(records: list[DraftRecord]) -> list[DraftRecord]:
    seen: set[str] = set()
    out: list[DraftRecord] = []
    for record in records:
        # T5 safe/danger pairs intentionally share the same utterance.
        if record.utterance_type == "T5":
            out.append(record)
            continue
        key = record.utterance.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def apply_post_filters(records: list[DraftRecord]) -> list[DraftRecord]:
    deduped = dedupe_records(records)
    for record in deduped:
        flags: list[str] = list(record.flags)
        if check_unrecoverable_args(record.utterance, record.slots):
            if "unrecoverable_arg" not in flags:
                flags.append("unrecoverable_arg")
        record.flags = flags
    return deduped


def collect_tool_names_from_records(records: list[DraftRecord]) -> set[str]:
    names: set[str] = set()
    for record in records:
        for call in record.expected.calls:
            if tool := call.get("tool"):
                names.add(tool)
    return names
