
import pandas as pd
import pytest

from bet.simulation import (
    Battery,
    simulate_factory,
    solar_pv_power,
    wind_turbine_power,
)


def test_factory_generates_measurements():
    data = simulate_factory(periods=96)

    assert not data.empty
    assert set(data["measurement_type"]).issuperset(
        {"power_kw", "energy_kwh"}
    )


def test_factory_is_reproducible():
    first = simulate_factory(periods=96, seed=42)
    second = simulate_factory(periods=96, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_factory_generates_expected_timestamps():
    data = simulate_factory(periods=96)

    assert data["timestamp_utc"].nunique() == 96


def test_solar_output_is_zero_without_irradiance():
    assert solar_pv_power(0, 25, 200) == 0


def test_solar_output_is_non_negative():
    assert solar_pv_power(800, 25, 200) >= 0


def test_wind_output_is_zero_below_cut_in():
    assert wind_turbine_power(2, 100) == 0


def test_wind_output_is_zero_above_cut_out():
    assert wind_turbine_power(26, 100) == 0


def test_battery_soc_stays_within_limits():
    battery = Battery()

    for _ in range(100):
        battery.dispatch(net_load_kw=100, interval_hours=0.25)

    assert battery.min_soc_pct <= battery.soc_pct <= battery.max_soc_pct


def test_invalid_periods_are_rejected():
    with pytest.raises((ValueError, TypeError)):
        simulate_factory(periods=0)


def test_invalid_initial_soc_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        Battery(initial_soc_pct=110)
