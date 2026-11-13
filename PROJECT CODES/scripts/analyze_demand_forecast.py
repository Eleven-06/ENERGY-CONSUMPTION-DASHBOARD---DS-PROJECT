
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bet.industrial_forecasting.evaluate import calculate_metrics

RESULTS = "models/industrial/demand_forecast_results.csv"


def main():
    data = pd.read_csv(RESULTS, parse_dates=["timestamp"])

    actual = data["actual_demand_kw"]
    predicted = data["xgboost_prediction_kw"]
    baseline = data["baseline_prediction_kw"]
    print("Baseline metrics:", calculate_metrics(actual, baseline))

    metrics = calculate_metrics(actual, predicted)

    print("\nFORECAST ERROR ANALYSIS")
    print("=" * 45)

    print("Maximum actual demand:", actual.max())
    print("Minimum actual demand:", actual.min())
    print("Maximum absolute error:", np.abs(actual - predicted).max())

    print("\nMetrics:", metrics)

    # Actual vs predicted demand
    plt.figure(figsize=(14, 5))
    plt.plot(data["timestamp"], actual, label="Actual")
    plt.plot(data["timestamp"], predicted, label="XGBoost")
    plt.title("Industrial Demand: Actual vs Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/industrial/demand_forecast_plot.png")
    plt.show()

    # Prediction residuals
    residuals = actual - predicted

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)
    plt.title("XGBoost Forecast Residual Distribution")
    plt.xlabel("Actual - Predicted (kW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("models/industrial/demand_residuals.png")
    plt.show()

    # Feature importance
    import joblib

    artifact = joblib.load(
        "models/industrial/industrial_demand_xgboost.joblib"
    )

    importance = pd.Series(
        artifact["model"].feature_importances_,
        index=artifact["features"],
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    importance.tail(15).plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/industrial/demand_feature_importance.png")
    plt.show()


if __name__ == "__main__":
    main()
