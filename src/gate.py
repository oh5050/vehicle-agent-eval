"""Gate: tier-2 tool eligibility checks against vehicle state.

Danger rule: speed_kmh > 0 OR gear in {"D", "R"}
When dangerous AND call targets a risk_tier=2 tool AND current kind is "execute",
demote kind to "confirm" and record a gate event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.registry import Registry


def is_dangerous_state(vehicle_state: dict[str, Any]) -> bool:
    """Return True when the vehicle is in a state that requires tier-2 confirmation."""
    return vehicle_state.get("speed_kmh", 0) > 0 or vehicle_state.get("gear") in ("D", "R")


def apply_gate(
    kind: str,
    calls: list[dict[str, Any]],
    vehicle_state: dict[str, Any],
    registry: "Registry",
) -> tuple[str, list[dict[str, Any]]]:
    """Apply tier-2 safety gate to a parsed model response.

    Args:
        kind: Original kind string from parsed response.
        calls: Parsed call list from the model response.
        vehicle_state: Current vehicle state dict.
        registry: Loaded Registry — used to look up risk_tier per tool.

    Returns:
        (new_kind, gate_events) where gate_events is a list of dicts describing
        each demotion that occurred.  If no demotion happened, gate_events is [].
    """
    if kind != "execute":
        return kind, []

    if not is_dangerous_state(vehicle_state):
        return kind, []

    tier2_names = {t.name for t in registry.tools_with_risk_tier(2)}
    gate_events: list[dict[str, Any]] = []

    for call in calls:
        tool = call.get("tool", "")
        if tool in tier2_names:
            gate_events.append(
                {
                    "tool": tool,
                    "reason": "tier-2 tool called while vehicle is in dangerous state",
                    "original_kind": "execute",
                    "new_kind": "confirm",
                    "speed_kmh": vehicle_state.get("speed_kmh"),
                    "gear": vehicle_state.get("gear"),
                }
            )

    if gate_events:
        return "confirm", gate_events
    return kind, []
