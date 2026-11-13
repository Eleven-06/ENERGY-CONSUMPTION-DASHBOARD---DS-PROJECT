
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bet.industrial.schema import EnergyReading
from bet.simulation.battery import Battery
from bet.simulation.machines import Machine, simulate_machine_power
from bet.simulation.solar_pv import solar_pv_power
from bet.simulation.wind_turbine import wind_turbine_power

DEFAULT_MACHINES = [
    Machine("CNC-001", 45.0, 0.90),
    Machine("MOTOR-001", 30.0, 0.95),
    Machine("COMPRESSOR-001", 55.0, 0.85),
    Machine("HVAC-001", 40.0, 0.90),
    Machine("PUMP-001", 22.0, 0.85),
    Machine("FURNACE-001", 80.0, 0.80),
]


def simulate_factory(
    start: str | datetime = "2026-01-01T00:00:00+00:00",
    periods: int = 96,
    interval_minutes: int = 15,
    seed: int = 42,
    solar_capacity_kw: float = 200.0,
    wind_rated_power_kw: float = 100.0,
    machines: list[Machine] | None = None,
    battery: Battery | None = None,
) -> pd.DataFrame:
    """
    Simulate industrial energy measurements.

    Returns a long-format DataFrame compatible with EnergyReading.
    """

    if periods <= 0:
        raise ValueError("periods must be positive")

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    if solar_capacity_kw <= 0 or wind_rated_power_kw <= 0:
        raise ValueError("Renewable capacities must be positive")

    rng = np.random.default_rng(seed)

    machine_list = (
        list(DEFAULT_MACHINES) if machines is None else list(machines)
    )

    if not machine_list:
        raise ValueError("At least one machine is required")

    if len({m.asset_id for m in machine_list}) != len(machine_list):
        raise ValueError("Machine asset IDs must be unique")

    battery = battery if battery is not None else Battery()

    start_time = pd.Timestamp(start)

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    timestamps = pd.date_range(
        start=start_time,
        periods=periods,
        freq=f"{interval_minutes}min",
    )

    interval_hours = interval_minutes / 60.0
    records = []

    def add_reading(
        timestamp,
        asset_id,
        measurement_type,
        value,
        unit,
    ):
        reading = EnergyReading(
            timestamp_utc=timestamp.to_pydatetime(),
            facility_id="FACTORY-001",
            asset_id=asset_id,
            measurement_type=measurement_type,
            value=float(value),
            unit=unit,
            source_type="simulated",
            quality_flag="good",
            dataset_version="1.0",
        )
        records.append(reading.to_dict())

    def add_power(timestamp, asset_id, power_kw):
        add_reading(
            timestamp, asset_id, "power_kw", power_kw, "kW"
        )
        add_reading(
            timestamp,
            asset_id,
            "energy_kwh",
            power_kw * interval_hours,
            "kWh",
        )

    for timestamp in timestamps:
        hour = timestamp.hour + timestamp.minute / 60.0

        # Synthetic weather drivers.
        daylight = max(
            0.0,
            np.sin(np.pi * (hour - 6.0) / 12.0),
        )

        cloud_factor = rng.uniform(0.70, 1.0)

        irradiance = 850.0 * daylight * cloud_factor

        ambient_temp = (
            28.0
            + 5.0 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
            + rng.normal(0.0, 0.5)
        )

        wind_speed = max(
            0.0,
            7.0
            + 2.0 * np.sin(2.0 * np.pi * hour / 24.0)
            + rng.normal(0.0, 1.2),
        )

        add_reading(
            timestamp,
            "WEATHER-001",
            "irradiance_w_m2",
            irradiance,
            "W/m2",
        )

        add_reading(
            timestamp,
            "WEATHER-001",
            "temperature_c",
            ambient_temp,
            "degC",
        )

        add_reading(
            timestamp,
            "WEATHER-001",
            "wind_speed_m_s",
            wind_speed,
            "m/s",
        )

        # Industrial machine demand.
        machine_powers = []

        for machine in machine_list:
            power = simulate_machine_power(machine, hour, rng)
            machine_powers.append(power)
            add_power(timestamp, machine.asset_id, power)

        total_demand_kw = sum(machine_powers)

        add_power(timestamp, "FACILITY-TOTAL", total_demand_kw)

        # Renewable generation.
        solar_kw = solar_pv_power(
            irradiance,
            ambient_temp,
            solar_capacity_kw,
        )

        wind_kw = wind_turbine_power(
            wind_speed,
            wind_rated_power_kw,
        )

        add_power(timestamp, "SOLAR-PV-001", solar_kw)
        add_power(timestamp, "WIND-001", wind_kw)

        # Net load before battery dispatch.
        renewable_kw = solar_kw + wind_kw
        net_load_kw = total_demand_kw - renewable_kw

        charge_kw, discharge_kw, soc_pct = battery.dispatch(
            net_load_kw,
            interval_hours,
        )

        # Grid imports and exports after battery operation.
        grid_import_kw = max(
            0.0,
            net_load_kw - discharge_kw,
        )

        grid_export_kw = max(
            0.0,
            -net_load_kw - charge_kw,
        )

        add_power(timestamp, "BATTERY-CHARGE", charge_kw)
        add_power(timestamp, "BATTERY-DISCHARGE", discharge_kw)

        add_reading(
            timestamp,
            "BATTERY-001",
            "battery_soc_pct",
            soc_pct,
            "%",
        )

        add_power(timestamp, "GRID-IMPORT", grid_import_kw)
        add_power(timestamp, "GRID-EXPORT", grid_export_kw)

    return pd.DataFrame(records)


if __name__ == "__main__":
    output_path = Path("data/simulated/industrial_factory_15min.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = simulate_factory(periods=96 * 30)

    dataset.to_csv(output_path, index=False)

    print(f"Generated {len(dataset)} measurements")
    print(f"Saved dataset: {output_path}")
    print(f"Time range: {dataset['timestamp_utc'].min()}")
    print(f"Time range: {dataset['timestamp_utc'].max()}")
