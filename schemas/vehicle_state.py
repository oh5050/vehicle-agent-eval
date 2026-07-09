"""Vehicle state schema for tier-2 gate checks.

Field set mirrors the context schema handed off by the data spec exactly;
gate.py should request an explicit spec update before adding fields here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Passengers(BaseModel):
    """Occupancy flags used by seat/climate gating."""

    front: bool
    rear: bool


class VehicleState(BaseModel):
    """Snapshot of vehicle conditions relevant to tool eligibility."""

    speed_kmh: float
    gear: Literal["P", "R", "N", "D"]
    battery_pct: float
    outside_temp_c: float
    passengers: Passengers
