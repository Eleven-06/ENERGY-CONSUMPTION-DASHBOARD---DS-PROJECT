from pathlib import Path

import pandas as pd

from bet.anomaly.forecast_residuals import ForecastResidualDetector

MODEL_DIR = Path("models/industrial")
OUTPUT_DIR = Path("results/anomaly")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_forecast_results(filename, actual_column, residual_name):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Forecast results not found: {path}")

    data = pd.read_csv(path)

    required = {
        "timestamp",
        actual_column,
        "xgboost_prediction_kw",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="raise",
    )

    if data["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps found in {filename}")

    data[residual_name] = (
        data[actual_column] - data["xgboost_prediction_kw"]
    )

    return data[["timestamp", residual_name]]


def main():
    demand = load_forecast_results(
        "demand_forecast_results.csv",
        "actual_demand_kw",
        "demand_residual_kw",
    )

    solar = load_forecast_results(
        "solar_forecast_results.csv",
        "actual_solar_power_kw",
        "solar_residual_kw",
    )

    wind = load_forecast_results(
        "wind_forecast_results.csv",
        "actual_wind_power_kw",
        "wind_residual_kw",
    )

    # Inner join ensures that all three residuals refer
    # to exactly the same forecast timestamp.
    combined = demand.merge(solar, on="timestamp", how="inner")
    combined = combined.merge(wind, on="timestamp", how="inner")

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if combined.empty:
        raise ValueError("No common timestamps across forecast results")

    residual_columns = [
        "demand_residual_kw",
        "solar_residual_kw",
        "wind_residual_kw",
    ]

    if combined[residual_columns].isna().any().any():
        raise ValueError("Missing residual values found")

    # Chronological calibration/detection split.
    split_index = int(len(combined) * 0.40)

    calibration_data = combined.iloc[:split_index].copy()
    detection_data = combined.iloc[split_index:].copy()

    detector = ForecastResidualDetector()

    detector.fit(calibration_data)
    results = detector.detect(detection_data)

    print("Calibration samples:", len(calibration_data))
    print("Detection samples:", len(detection_data))

    output_path = OUTPUT_DIR / "forecast_residual_anomalies.csv"
    results.to_csv(output_path, index=False)

    print("\nFORECAST RESIDUAL ANOMALY DETECTION")
    print("=" * 55)

    print("Common timestamps:", len(results))
    print("Detected anomalies:", int(results["is_anomaly"].sum()))

    print("\nResidual thresholds:")

    for name, threshold in detector.thresholds.items():
        print(
            f"{name}: "
            f"median={threshold.median:.4f} kW, "
            f"MAD={threshold.mad:.4f} kW, "
            f"threshold={threshold.threshold:.4f} kW"
        )

    print("\nAnomaly counts by source:")

    for column in [
        "demand_anomaly",
        "solar_anomaly",
        "wind_anomaly",
    ]:
        print(f"{column}: {int(results[column].sum())}")

    print("\nMost unusual timestamps:")

    unusual = results[results["is_anomaly"]].copy()

    unusual["total_absolute_residual_kw"] = (
        unusual["demand_absolute_residual_kw"]
        + unusual["solar_absolute_residual_kw"]
        + unusual["wind_absolute_residual_kw"]
    )

    print(
        unusual.sort_values(
            "total_absolute_residual_kw",
            ascending=False,
        )[
            [
                "timestamp",
                "demand_residual_kw",
                "solar_residual_kw",
                "wind_residual_kw",
                "anomaly_sources",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()