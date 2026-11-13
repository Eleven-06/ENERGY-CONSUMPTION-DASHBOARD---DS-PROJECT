
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FaultEvent:
    fault_type: str
    asset_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    affected_timestamps: tuple[pd.Timestamp, ...]


FAULT_SCENARIOS = [
    {
        "fault_type": "demand_spike",
        "asset_id": "FACILITY-TOTAL",
        "column": "FACILITY-TOTAL__power_kw",
        "operation": "multiply",
        "factor": 1.8,
    },
    {
        "fault_type": "demand_drop",
        "asset_id": "FACILITY-TOTAL",
        "column": "FACILITY-TOTAL__power_kw",
        "operation": "multiply",
        "factor": 0.3,
    },
    {
        "fault_type": "solar_degradation",
        "asset_id": "SOLAR-PV-001",
        "column": "SOLAR-PV-001__power_kw",
        "operation": "multiply",
        "factor": 0.2,
    },
    {
        "fault_type": "wind_generation_loss",
        "asset_id": "WIND-001",
        "column": "WIND-001__power_kw",
        "operation": "zero",
    },
    {
        "fault_type": "grid_import_disturbance",
        "asset_id": "GRID-IMPORT",
        "column": "GRID-IMPORT__power_kw",
        "operation": "add",
        "value": 50.0,
    },
    {
        "fault_type": "wind_speed_sensor_stuck",
        "asset_id": "WEATHER-001",
        "column": "WEATHER-001__wind_speed_m_s",
        "operation": "constant",
    },
]


def inject_industrial_faults(
    data: pd.DataFrame,
    duration_intervals: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FaultEvent]]:
    """
    Inject deterministic faults into a copy of an industrial time series.

    Returns:
        modified_data: fault-injected copy
        labels: timestamp-level ground truth
        events: metadata for each injected event
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data must have a DatetimeIndex")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically")

    if data.index.has_duplicates:
        raise ValueError("Data index must not contain duplicate timestamps")

    if duration_intervals < 1:
        raise ValueError("duration_intervals must be at least 1")

    missing = [
        scenario["column"]
        for scenario in FAULT_SCENARIOS
        if scenario["column"] not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {sorted(set(missing))}")

    modified = data.copy(deep=True)

    labels = pd.DataFrame(
        {
            "timestamp": modified.index,
            "is_fault": False,
            "fault_type": "",
            "asset_id": "",
        }
    )

    labels = labels.set_index("timestamp")
    events = []

    # Spread events across the detection period to avoid overlap.
    fractions = [0.08, 0.23, 0.38, 0.53, 0.68, 0.83]
    total = len(modified)

    if total < len(FAULT_SCENARIOS) * duration_intervals:
        raise ValueError("Detection data is too short for all fault scenarios")

    for scenario, fraction in zip(FAULT_SCENARIOS, fractions, strict=False):
        start_position = int(total * fraction)
        start_position = min(
            start_position,
            total - duration_intervals,
        )

        positions = range(
            start_position,
            start_position + duration_intervals,
        )

        timestamps = tuple(modified.index[position] for position in positions)
        column = scenario["column"]
        original_values = modified.iloc[list(positions)][column].copy()

        if scenario["operation"] == "multiply":
            modified.loc[list(timestamps), column] = (
                original_values.to_numpy() * scenario["factor"]
            )

        elif scenario["operation"] == "zero":
            modified.loc[list(timestamps), column] = 0.0

        elif scenario["operation"] == "add":
            modified.loc[list(timestamps), column] = (
                original_values.to_numpy() + scenario["value"]
            )

        elif scenario["operation"] == "constant":
            modified.loc[list(timestamps), column] = float(
                original_values.iloc[0]
            )

        labels.loc[list(timestamps), "is_fault"] = True
        labels.loc[list(timestamps), "fault_type"] = scenario["fault_type"]
        labels.loc[list(timestamps), "asset_id"] = scenario["asset_id"]

        events.append(
            FaultEvent(
                fault_type=scenario["fault_type"],
                asset_id=scenario["asset_id"],
                start=timestamps[0],
                end=timestamps[-1],
                affected_timestamps=timestamps,
            )
        )

    return modified, labels.reset_index(), events
