
from dataclasses import dataclass

import numpy as np


@dataclass
class Battery:
    capacity_kwh: float = 500.0
    max_power_kw: float = 150.0
    initial_soc_pct: float = 50.0
    min_soc_pct: float = 10.0
    max_soc_pct: float = 95.0
    efficiency: float = 0.95

    def __post_init__(self):
        if self.capacity_kwh <= 0 or self.max_power_kw <= 0:
            raise ValueError("Battery capacity and power must be positive")

        if not 0 <= self.min_soc_pct < self.max_soc_pct <= 100:
            raise ValueError("Invalid battery SOC limits")

        if not self.min_soc_pct <= self.initial_soc_pct <= self.max_soc_pct:
            raise ValueError("Initial SOC must be within battery limits")

        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")

        self.energy_kwh = (
            self.capacity_kwh * self.initial_soc_pct / 100.0
        )

    @property
    def soc_pct(self) -> float:
        return 100.0 * self.energy_kwh / self.capacity_kwh

    def dispatch(self, net_load_kw: float, interval_hours: float):
        """
        Positive net load: discharge to cover demand.
        Negative net load: charge using renewable surplus.

        Returns charge power, discharge power and SOC.
        """

        if interval_hours <= 0:
            raise ValueError("interval_hours must be positive")
        if not np.isfinite(net_load_kw):
            raise ValueError("net_load_kw must be finite")

        if not np.isfinite(interval_hours):
            raise ValueError("interval_hours must be finite")

        charge_kw = 0.0
        discharge_kw = 0.0

        min_energy = self.capacity_kwh * self.min_soc_pct / 100.0
        max_energy = self.capacity_kwh * self.max_soc_pct / 100.0

        if net_load_kw > 0:
            available_energy = max(
                0.0,
                self.energy_kwh - min_energy,
            )

            max_discharge_by_energy = (
                available_energy * self.efficiency / interval_hours
            )

            discharge_kw = min(
                net_load_kw,
                self.max_power_kw,
                max_discharge_by_energy,
            )

            self.energy_kwh -= (
                discharge_kw * interval_hours / self.efficiency
            )

        elif net_load_kw < 0:
            available_capacity = max(
                0.0,
                max_energy - self.energy_kwh,
            )

            max_charge_by_capacity = (
                available_capacity / (self.efficiency * interval_hours)
            )

            charge_kw = min(
                -net_load_kw,
                self.max_power_kw,
                max_charge_by_capacity,
            )

            self.energy_kwh += (
                charge_kw * interval_hours * self.efficiency
            )

        self.energy_kwh = min(
            max(self.energy_kwh, min_energy),
            max_energy,
        )

        return (
            round(charge_kw, 3),
            round(discharge_kw, 3),
            round(self.soc_pct, 3),
        )
