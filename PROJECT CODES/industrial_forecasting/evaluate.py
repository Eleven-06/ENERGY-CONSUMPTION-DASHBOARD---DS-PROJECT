
import numpy as np
import pandas as pd


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have equal non-zero length"
        )

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE is undefined when actual demand is zero.
    nonzero = np.abs(actual) > 1e-8

    mape = (
        np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_pct": float(mape),
    }


def evaluate_forecasts(results: pd.DataFrame):
    required_columns = [
        "actual_demand_kw",
        "xgboost_prediction_kw",
        "baseline_prediction_kw",
    ]

    missing = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    actual = results["actual_demand_kw"].to_numpy(dtype=float)
    xgboost_predictions = results["xgboost_prediction_kw"].to_numpy(
        dtype=float
    )
    baseline_predictions = results["baseline_prediction_kw"].to_numpy(
        dtype=float
    )

    # Operating period: actual demand greater than zero.
    operating_mask = actual > 1e-8

    # Non-operating period: actual demand approximately zero.
    off_mask = ~operating_mask

    return {
        "Overall": {
            "XGBoost": calculate_metrics(actual, xgboost_predictions),
            "Persistence baseline": calculate_metrics(
                actual, baseline_predictions
            ),
        },
        "Operating period": {
            "sample_count": int(operating_mask.sum()),
            "XGBoost": calculate_metrics(
                actual[operating_mask],
                xgboost_predictions[operating_mask],
            ),
            "Persistence baseline": calculate_metrics(
                actual[operating_mask],
                baseline_predictions[operating_mask],
            ),
        },
        "Non-operating period": {
            "sample_count": int(off_mask.sum()),
            "XGBoost": calculate_metrics(
                actual[off_mask],
                xgboost_predictions[off_mask],
            ),
            "Persistence baseline": calculate_metrics(
                actual[off_mask],
                baseline_predictions[off_mask],
            ),
        },
    }
