
import pandas as pd

from bet.analytics.aggregation import load_energy_timeseries


def prepare_demand_dataset(data=None):
    """
    Predict demand at t+1 using information available at t.

    Each row's index is the forecast origin timestamp.
    The target is the next 15-minute demand.
    """
    if data is None:
        data = load_energy_timeseries()

    target_column = "FACILITY-TOTAL__power_kw"

    if target_column not in data.columns:
        raise ValueError(f"Missing measurement: {target_column}")

    demand = data[target_column].astype(float)

    df = pd.DataFrame(index=data.index)

    # Target: next observation (15 minutes ahead)
    df["target_demand_kw"] = demand.shift(-1)

    # Latest observed demand is lag 1 for the next interval
    for lag in [1, 2, 4, 8, 96]:
        df[f"lag_{lag}"] = demand.shift(lag - 1)

    # Rolling statistics include current and past observations only
    for window in [4, 8, 24, 96]:
        history = demand.rolling(window)

        df[f"rolling_mean_{window}"] = history.mean()
        df[f"rolling_std_{window}"] = history.std()

    # Calendar features describe the target timestamp
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
        raise ValueError("Not enough data to create forecasting features")

    return df


def chronological_split(df, test_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    split_index = int(len(df) * (1 - test_fraction))

    if split_index == 0 or split_index == len(df):
        raise ValueError("Insufficient data for chronological split")

    return (
        df.iloc[:split_index].copy(),
        df.iloc[split_index:].copy(),
    )
