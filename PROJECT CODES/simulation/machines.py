
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Machine:
    asset_id: str
    rated_power_kw: float
    operating_probability: float = 0.85
    start_hour: int = 6
    end_hour: int = 22

    def __post_init__(self):
        if not self.asset_id.strip():
            raise ValueError("asset_id cannot be empty")
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")
        if not 0 <= self.operating_probability <= 1:
            raise ValueError("operating_probability must be between 0 and 1")
        if not 0 <= self.start_hour <= 23:
            raise ValueError("start_hour must be between 0 and 23")
        if not 1 <= self.end_hour <= 24:
            raise ValueError("end_hour must be between 1 and 24")
        if self.start_hour >= self.end_hour:
            raise ValueError("start_hour must be earlier than end_hour")


def simulate_machine_power(
    machine: Machine,
    hour: float,
    rng: np.random.Generator,
) -> float:
    """Generate one machine's power demand in kW."""

    if not machine.start_hour <= hour < machine.end_hour:
        return 0.0

    if rng.random() > machine.operating_probability:
        return 0.0

    load_factor = rng.uniform(0.60, 1.0)
    noise = rng.normal(1.0, 0.025)

    return round(
        machine.rated_power_kw * load_factor * max(noise, 0.0),
        3,
    )
