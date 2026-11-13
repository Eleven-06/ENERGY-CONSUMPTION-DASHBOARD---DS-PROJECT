
import pandas as pd

from bet.analytics.aggregation import load_energy_timeseries


def prepare_wind_dataset(data=None):
    """
    Predict wind turbine power generation at t+1.

    Forecast horizon: 15 minutes.
    Features use current and historical observations only.
    """

    if data is None:
        data = load_energy_timeseries()

    wind_column = "WIND-001__power_kw"
    speed_column = "WEATHER-001__wind_speed_m_s"

    required_columns = [wind_column, speed_column]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.sort_index()

    wind = data[wind_column].astype(float)
    wind_speed = data[speed_column].astype(float)

    df = pd.DataFrame(index=data.index)

    # Next 15-minute wind power is the target.
    df["target_wind_power_kw"] = wind.shift(-1)

    # Historical wind power.
    for lag in [1, 2, 4, 8, 96]:
        df[f"wind_lag_{lag}"] = wind.shift(lag - 1)

    # Rolling wind power statistics.
    for window in [4, 8, 24, 96]:
        history = wind.rolling(window)

        df[f"wind_rolling_mean_{window}"] = history.mean()
        df[f"wind_rolling_std_{window}"] = history.std()

    # Current and historical wind speed.
    df["wind_speed_current"] = wind_speed

    for lag in [1, 2, 4, 8]:
        df[f"wind_speed_lag_{lag}"] = wind_speed.shift(lag)

    # Rolling wind speed statistics.
    for window in [4, 8, 24]:
        df[f"wind_speed_rolling_mean_{window}"] = (
            wind_speed.rolling(window).mean()
        )

        df[f"wind_speed_rolling_std_{window}"] = (
            wind_speed.rolling(window).std()
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
            "Not enough data to create wind forecasting features"
        )

    return df
