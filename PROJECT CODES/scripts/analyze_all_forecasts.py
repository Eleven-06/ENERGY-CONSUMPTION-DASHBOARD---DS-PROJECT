
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = Path("models/industrial")
OUTPUT_DIR = Path("results/forecasting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_CONFIG = {
    "Demand": {
        "results": "demand_forecast_results.csv",
        "model": "industrial_demand_xgboost.joblib",
        "actual": "actual_demand_kw",
        "prediction": "xgboost_prediction_kw",
        "baseline": "baseline_prediction_kw",
        "unit": "kW",
    },
    "Solar": {
        "results": "solar_forecast_results.csv",
        "model": "industrial_solar_xgboost.joblib",
        "actual": "actual_solar_power_kw",
        "prediction": "xgboost_prediction_kw",
        "baseline": "baseline_prediction_kw",
        "unit": "kW",
    },
    "Wind": {
        "results": "wind_forecast_results.csv",
        "model": "industrial_wind_xgboost.joblib",
        "actual": "actual_wind_power_kw",
        "prediction": "xgboost_prediction_kw",
        "baseline": "baseline_prediction_kw",
        "unit": "kW",
    },
}


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = np.abs(actual) > 1e-8

    mape = (
        np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )

    # Uses maximum observed actual value as a common
    # dataset-specific reference, not rated capacity.
    reference = np.max(np.abs(actual))

    nmae = (
        (mae / reference) * 100
        if reference > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
        "NMAE_observed_max_pct": nmae,
    }


def load_forecasts():
    forecasts = {}

    for name, config in MODEL_CONFIG.items():
        results_path = MODEL_DIR / config["results"]

        if not results_path.exists():
            raise FileNotFoundError(results_path)

        df = pd.read_csv(results_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        required = [
            config["actual"],
            config["prediction"],
            config["baseline"],
        ]

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"{name} results missing columns: {missing}"
            )

        forecasts[name] = df

    return forecasts


def plot_actual_vs_predicted(name, df, config):
    plt.figure(figsize=(14, 5))

    plt.plot(
        df["timestamp"],
        df[config["actual"]],
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        df["timestamp"],
        df[config["prediction"]],
        label="XGBoost prediction",
        linewidth=1.3,
    )

    plt.xlabel("Timestamp")
    plt.ylabel(f"Power ({config['unit']})")
    plt.title(f"{name}: Actual vs Predicted")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = OUTPUT_DIR / f"{name.lower()}_actual_vs_predicted.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print("Saved:", path)


def plot_residuals(name, df, config):
    actual = df[config["actual"]].to_numpy()
    predicted = df[config["prediction"]].to_numpy()

    residuals = actual - predicted

    plt.figure(figsize=(9, 5))
    plt.hist(residuals, bins=40, edgecolor="black")

    plt.axvline(0, linestyle="--", linewidth=1.5)

    plt.xlabel(f"Residual ({config['unit']})")
    plt.ylabel("Frequency")
    plt.title(f"{name}: Forecast Residual Distribution")
    plt.tight_layout()

    path = OUTPUT_DIR / f"{name.lower()}_residuals.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print("Saved:", path)


def plot_model_comparison(metrics_df):
    metric_columns = ["MAE", "RMSE", "MAPE_pct"]

    for metric in metric_columns:
        plt.figure(figsize=(9, 5))

        metrics_df[[f"{metric}_XGBoost",
                    f"{metric}_Persistence"]].plot(
            kind="bar",
            ax=plt.gca(),
        )

        plt.title(f"XGBoost vs Persistence: {metric}")
        plt.ylabel(metric)
        plt.xlabel("Forecasting Model")
        plt.xticks(rotation=0)
        plt.tight_layout()

        path = OUTPUT_DIR / f"comparison_{metric.lower()}.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print("Saved:", path)


def plot_feature_importance(name):
    model_path = MODEL_DIR / MODEL_CONFIG[name]["model"]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["feature"][::-1],
        importance["importance"][::-1],
    )

    plt.xlabel("Feature importance")
    plt.title(f"{name}: Top 15 Important Features")
    plt.tight_layout()

    path = OUTPUT_DIR / f"{name.lower()}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print("Saved:", path)


def main():
    forecasts = load_forecasts()

    metric_rows = []

    print("\nUNIFIED FORECAST ANALYSIS")
    print("=" * 55)

    for name, df in forecasts.items():
        config = MODEL_CONFIG[name]

        actual = df[config["actual"]].to_numpy()
        predicted = df[config["prediction"]].to_numpy()
        baseline = df[config["baseline"]].to_numpy()

        xgb_metrics = calculate_metrics(actual, predicted)
        baseline_metrics = calculate_metrics(actual, baseline)

        metric_rows.append({
            "Model": name,
            **{f"{key}_XGBoost": value
               for key, value in xgb_metrics.items()},
            **{f"{key}_Persistence": value
               for key, value in baseline_metrics.items()},
        })

        print(f"\n{name} Forecast")
        print("-" * 40)
        print("Test samples:", len(df))

        for label, metrics in [
            ("XGBoost", xgb_metrics),
            ("Persistence", baseline_metrics),
        ]:
            print(f"\n{label}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

        plot_actual_vs_predicted(name, df, config)
        plot_residuals(name, df, config)
        plot_feature_importance(name)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = OUTPUT_DIR / "forecast_metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    plot_model_comparison(metrics_df)

    print("\nANALYSIS COMPLETED")
    print("=" * 55)
    print("Metrics saved:", metrics_path)
    print("All visualizations saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
