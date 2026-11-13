
import pandas as pd

from bet.analytics.aggregation import load_energy_timeseries


def prepare_solar_dataset(data=None):
    """
    Predict solar PV power generation at t+1 (15 minutes ahead).

    Features use only information available at forecast origin t.
    The target is the next 15-minute solar power measurement.
    """

    if data is None:
        data = load_energy_timeseries()

    solar_column = "SOLAR-PV-001__power_kw"
    irradiance_column = "WEATHER-001__irradiance_w_m2"
    temperature_column = "WEATHER-001__temperature_c"

    required_columns = [
        solar_column,
        irradiance_column,
        temperature_column,
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.sort_index()

    solar = data[solar_column].astype(float)
    irradiance = data[irradiance_column].astype(float)
    temperature = data[temperature_column].astype(float)

    df = pd.DataFrame(index=data.index)

    # Target: next 15-minute solar generation.
    df["target_solar_power_kw"] = solar.shift(-1)

    # Historical solar generation.
    for lag in [1, 2, 4, 8, 96]:
        df[f"solar_lag_{lag}"] = solar.shift(lag - 1)

    # Rolling solar generation statistics.
    for window in [4, 8, 24, 96]:
        history = solar.rolling(window)

        df[f"solar_rolling_mean_{window}"] = history.mean()
        df[f"solar_rolling_std_{window}"] = history.std()

    # Current and historical weather observations.
    df["irradiance_current"] = irradiance
    df["temperature_current"] = temperature

    for lag in [1, 2, 4]:
        df[f"irradiance_lag_{lag}"] = irradiance.shift(lag)

    for lag in [1, 2, 4]:
        df[f"temperature_lag_{lag}"] = temperature.shift(lag)

    # Rolling irradiance statistics.
    for window in [4, 8, 24]:
        df[f"irradiance_rolling_mean_{window}"] = (
            irradiance.rolling(window).mean()
        )

    # Calendar features for the target timestamp.
    target_time = df.index + pd.Timedelta(minutes=15)

    df["hour"] = target_time.hour
    df["minute"] = target_time.minute
    df["day_of_week"] = target_time.dayofweek
    df["day_of_month"] = target_time.day
    df["month"] = target_time.month
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Not enough data to create solar forecasting features"
        )

    return df
