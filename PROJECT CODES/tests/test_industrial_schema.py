
from datetime import UTC, datetime

import pytest

from bet.industrial.schema import EnergyReading


def create_reading(**overrides):
    values = {
        "timestamp_utc": datetime.now(UTC),
        "facility_id": "FACTORY-001",
        "asset_id": "CNC-001",
        "measurement_type": "power_kw",
        "value": 45.5,
        "unit": "kW",
        "source_type": "simulated",
        "quality_flag": "good",
        "dataset_version": "1.0",
    }
    values.update(overrides)
    return EnergyReading(**values)


def test_valid_industrial_reading():
    reading = create_reading()
    assert reading.facility_id == "FACTORY-001"
    assert reading.value == 45.5


def test_timezone_is_required():
    with pytest.raises(ValueError, match="timezone-aware"):
        create_reading(timestamp_utc=datetime(2026, 1, 1))


def test_negative_power_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        create_reading(value=-10)


def test_invalid_unit_is_rejected():
    with pytest.raises(ValueError, match="requires unit"):
        create_reading(unit="kWh")


def test_battery_soc_must_be_between_zero_and_hundred():
    with pytest.raises(ValueError, match="between 0 and 100"):
        create_reading(
            measurement_type="battery_soc_pct",
            value=120,
            unit="%",
        )


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError, match="Invalid source_type"):
        create_reading(source_type="unknown")


def test_temperature_can_be_negative():
    reading = create_reading(
        measurement_type="temperature_c",
        value=-5,
        unit="degC",
    )
    assert reading.value == -5


def test_reading_serialization_round_trip():
    original = create_reading()
    restored = EnergyReading.from_dict(original.to_dict())
    assert restored == original
