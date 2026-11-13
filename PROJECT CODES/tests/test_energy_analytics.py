
import pandas as pd
import pytest

from bet.analytics.energy_metrics import (
    calculate_daily_energy,
    calculate_energy_kpis,
)


@pytest.fixture
def sample_data():
    index = pd.date_range(
        "2026-01-01",
        periods=2,
        freq="15min",
        tz="UTC",
        name="timestamp_utc",
    )

    return pd.DataFrame(
        {
            "FACILITY-TOTAL__power_kw": [10, 30],
            "FACILITY-TOTAL__energy_kwh": [2.5, 7.5],

            "SOLAR-PV-001__power_kw": [1, 2],
            "SOLAR-PV-001__energy_kwh": [1, 1],

            "WIND-001__power_kw": [0, 1],
            "WIND-001__energy_kwh": [0.5, 0.5],

            "GRID-IMPORT__power_kw": [10, 26],
            "GRID-IMPORT__energy_kwh": [2, 6],

            "GRID-EXPORT__power_kw": [0, 0],
            "GRID-EXPORT__energy_kwh": [0, 0],

            "BATTERY-CHARGE__power_kw": [1, 0],
            "BATTERY-CHARGE__energy_kwh": [0.25, 0],

            "BATTERY-DISCHARGE__power_kw": [0, 1],
            "BATTERY-DISCHARGE__energy_kwh": [0, 0.25],

            "BATTERY-001__battery_soc_pct": [50, 40],
        },
        index=index,
    )


def test_facility_energy_kpis(sample_data):
    result = calculate_energy_kpis(sample_data)

    assert result["facility_energy_kwh"] == pytest.approx(10)
    assert result["facility_average_power_kw"] == pytest.approx(20)
    assert result["facility_peak_power_kw"] == pytest.approx(30)


def test_renewable_generation_and_share(sample_data):
    result = calculate_energy_kpis(sample_data)

    assert result["renewable_energy_kwh"] == pytest.approx(3)
    assert result["renewable_share_pct"] == pytest.approx(30)


def test_battery_metrics(sample_data):
    result = calculate_energy_kpis(sample_data)

    assert result["battery_charge_energy_kwh"] == pytest.approx(0.25)
    assert result["battery_discharge_energy_kwh"] == pytest.approx(0.25)
    assert result["battery_average_soc_pct"] == pytest.approx(45)


def test_power_balance(sample_data):
    result = calculate_energy_kpis(sample_data)

    assert result["power_balance_mean_residual_kw"] == pytest.approx(0)
    assert result["power_balance_max_abs_residual_kw"] == pytest.approx(0)


def test_daily_energy_aggregation(sample_data):
    result = calculate_daily_energy(sample_data)

    assert len(result) == 1
    assert result.iloc[0]["facility_demand_kwh"] == pytest.approx(10)
    assert result.iloc[0]["solar_generation_kwh"] == pytest.approx(2)
    assert result.iloc[0]["wind_generation_kwh"] == pytest.approx(1)
