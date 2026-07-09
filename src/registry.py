"""Tool registry: load, validate, and derive variants of the tool schemas.

Schema shape (matches schemas/tools_core.json, tools_tier2.json,
tools_distractor.json):

    {
      "name": str,
      "risk_tier": int,
      "description": str,
      "parameters": {
        "<param_name>": {
          "type": str | omitted,   # tier2 entries may omit "type" entirely
          "enum": [...] | omitted,
          "minimum"/"maximum": number | omitted,
          "default": Any | omitted,
          "unit": str | omitted,
          "description": str | omitted,
          "pattern": str | omitted,       # regex; violation → SCHEMA_VIOLATION
          "items": {"type": str | omitted, "enum": [...] | omitted}  # for arrays
        },
        ...
      },
      "required": [str, ...]          # all listed params must be present
      "required_one_of": [str, ...]   # at least one of these must be present
    }

Registry variants:
    - "core8":      tools_core.json (6) + tools_tier2.json (2) = 8 tools.
    - "core8_free": core8 with every "enum" removed and folded into the
                    parameter's "description" instead (E2 ablation). Derived
                    purely in memory; source JSON files are never written to.
    - "full18":     core8 + tools_distractor.json (10) = 18 tools (E3 ablation).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

RegistryVariant = Literal["core8", "core8_free", "full18"]

ErrorCode = Literal[
    "UNKNOWN_TOOL",
    "MISSING_REQUIRED",
    "SCHEMA_VIOLATION",
    "OUT_OF_RANGE",
]


class ArrayItemSchema(BaseModel):
    """Constraint on elements of an "array"-typed parameter."""

    type: str | None = None
    enum: list[Any] | None = None


class ParameterSchema(BaseModel):
    """Definition of a single tool parameter, as authored in schemas/*.json."""

    type: str | None = None
    description: str | None = None
    enum: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
    unit: str | None = None
    pattern: str | None = None
    items: ArrayItemSchema | None = None


class ToolSchema(BaseModel):
    """A single tool/function definition."""

    name: str
    description: str
    risk_tier: int | None = None
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    required_one_of: list[str] | None = None


class ValidationResult(BaseModel):
    """Outcome of validate_call: pass, or exactly one categorized error."""

    valid: bool
    tool_name: str
    error_code: ErrorCode | None = None
    error_message: str | None = None
    field: str | None = None


def _load_json_tools(path: Path) -> list[ToolSchema]:
    """Read a schemas/*.json file and parse it into ToolSchema objects.

    Returns an empty list if the file does not exist or holds an empty array
    (e.g. tools_distractor.json before the E3 spec is finalized).
    """
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return []
    try:
        return [ToolSchema.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"Invalid tool schema in {path}: {exc}") from exc


def _strip_enums_to_description(tool: ToolSchema) -> ToolSchema:
    """Return a deep copy of tool with every "enum" removed and folded into
    that parameter's "description" as guidance text.

    Never mutates the input tool or any on-disk file — operates entirely on
    an in-memory deep copy, per the E2 ablation requirement.
    """
    new_tool = tool.model_copy(deep=True)
    for spec in new_tool.parameters.values():
        if spec.enum is None:
            continue
        allowed = ", ".join(str(v) for v in spec.enum)
        guidance = f"허용값 중 하나를 정확히 사용하세요: {allowed}."
        spec.description = f"{spec.description} {guidance}".strip() if spec.description else guidance
        spec.enum = None
    return new_tool


def _merge_tools(*groups: list[ToolSchema]) -> list[ToolSchema]:
    """Concatenate tool groups, de-duplicating by name (first occurrence wins).

    Group order encodes precedence: core8 tools always win over distractor
    tools that reuse the same name.
    """
    merged: list[ToolSchema] = []
    seen: set[str] = set()
    for group in groups:
        for tool in group:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            merged.append(tool)
    return merged


def _type_matches(value: Any, expected_type: str) -> bool:
    """Check a Python value against a JSON-schema-style type name."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


class Registry:
    """A loaded, immutable set of tool schemas plus call validation."""

    def __init__(self, variant: RegistryVariant, tools: list[ToolSchema]):
        self.variant = variant
        self.tools = tools
        self._by_name = {t.name: t for t in tools}

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get_tool(self, tool_name: str) -> ToolSchema | None:
        return self._by_name.get(tool_name)

    def tools_with_risk_tier(self, risk_tier: int) -> list[ToolSchema]:
        """Return tools whose ``risk_tier`` metadata equals *risk_tier*.

        Gate checks must key off this metadata, not on which JSON file a tool
        came from (e.g. ``message_send`` lives in tools_distractor.json but
        has ``risk_tier=2`` and must be gated like other tier-2 tools).
        """
        return [t for t in self.tools if t.risk_tier == risk_tier]

    def validate_call(self, tool_name: str, args: dict[str, Any]) -> ValidationResult:
        """Validate a proposed tool call against this registry.

        Checks run in this order, returning on the first violation found:
          1. UNKNOWN_TOOL      — tool_name not present in this registry
          2. MISSING_REQUIRED  — a "required" field is absent, or none of
                                  "required_one_of" is present
          3. SCHEMA_VIOLATION  — an arg is not a declared parameter, its
                                  value's type does not match, or a string
                                  fails the declared regex pattern
          4. OUT_OF_RANGE      — value fails enum membership, numeric
                                  minimum/maximum, or array-item enum
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return ValidationResult(
                valid=False,
                tool_name=tool_name,
                error_code="UNKNOWN_TOOL",
                error_message=f"'{tool_name}' is not a registered tool in variant '{self.variant}'.",
            )

        for field_name in tool.required:
            if field_name not in args:
                return ValidationResult(
                    valid=False,
                    tool_name=tool_name,
                    error_code="MISSING_REQUIRED",
                    error_message=f"Missing required field '{field_name}'.",
                    field=field_name,
                )

        if tool.required_one_of and not any(f in args for f in tool.required_one_of):
            return ValidationResult(
                valid=False,
                tool_name=tool_name,
                error_code="MISSING_REQUIRED",
                error_message=f"At least one of {tool.required_one_of} is required.",
            )

        for field_name, value in args.items():
            spec = tool.parameters.get(field_name)
            if spec is None:
                return ValidationResult(
                    valid=False,
                    tool_name=tool_name,
                    error_code="SCHEMA_VIOLATION",
                    error_message=f"'{field_name}' is not a declared parameter for tool '{tool_name}'.",
                    field=field_name,
                )

            if spec.type is not None and not _type_matches(value, spec.type):
                return ValidationResult(
                    valid=False,
                    tool_name=tool_name,
                    error_code="SCHEMA_VIOLATION",
                    error_message=f"'{field_name}' expected type '{spec.type}', got '{type(value).__name__}'.",
                    field=field_name,
                )

            if spec.pattern is not None and isinstance(value, str):
                if re.fullmatch(spec.pattern, value) is None:
                    return ValidationResult(
                        valid=False,
                        tool_name=tool_name,
                        error_code="SCHEMA_VIOLATION",
                        error_message=f"'{field_name}' value {value!r} does not match pattern {spec.pattern!r}.",
                        field=field_name,
                    )

            if spec.enum is not None and value not in spec.enum:
                return ValidationResult(
                    valid=False,
                    tool_name=tool_name,
                    error_code="OUT_OF_RANGE",
                    error_message=f"'{field_name}' value {value!r} is not among allowed values {spec.enum}.",
                    field=field_name,
                )

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if spec.minimum is not None and value < spec.minimum:
                    return ValidationResult(
                        valid=False,
                        tool_name=tool_name,
                        error_code="OUT_OF_RANGE",
                        error_message=f"'{field_name}' value {value} is below minimum {spec.minimum}.",
                        field=field_name,
                    )
                if spec.maximum is not None and value > spec.maximum:
                    return ValidationResult(
                        valid=False,
                        tool_name=tool_name,
                        error_code="OUT_OF_RANGE",
                        error_message=f"'{field_name}' value {value} exceeds maximum {spec.maximum}.",
                        field=field_name,
                    )

            if (
                spec.type == "array"
                and spec.items is not None
                and spec.items.enum is not None
                and isinstance(value, list)
            ):
                for element in value:
                    if element not in spec.items.enum:
                        return ValidationResult(
                            valid=False,
                            tool_name=tool_name,
                            error_code="OUT_OF_RANGE",
                            error_message=(
                                f"'{field_name}' contains {element!r}, "
                                f"not among allowed values {spec.items.enum}."
                            ),
                            field=field_name,
                        )

        return ValidationResult(valid=True, tool_name=tool_name)


def load_registry(
    variant: RegistryVariant,
    *,
    schemas_dir: Path = SCHEMAS_DIR,
) -> Registry:
    """Load a registry variant.

    - "core8":      tools_core.json + tools_tier2.json (6 + 2 = 8 tools).
    - "core8_free": core8 with enums stripped in memory (E2 variant).
    - "full18":     core8 + tools_distractor.json (10) = 18 tools (E3 ablation).
    """
    core = _load_json_tools(schemas_dir / "tools_core.json")
    tier2 = _load_json_tools(schemas_dir / "tools_tier2.json")
    core8 = _merge_tools(core, tier2)

    if variant == "core8":
        return Registry(variant, core8)

    if variant == "core8_free":
        free_tools = [_strip_enums_to_description(t) for t in core8]
        return Registry(variant, free_tools)

    if variant == "full18":
        distractor = _load_json_tools(schemas_dir / "tools_distractor.json")
        return Registry(variant, _merge_tools(core8, distractor))

    raise ValueError(f"Unknown registry variant: {variant!r}")
