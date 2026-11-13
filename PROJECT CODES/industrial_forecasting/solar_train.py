
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from bet.industrial_forecasting.solar_dataset import (
    prepare_solar_dataset,
)

MODEL_DIR = Path("models/industrial")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual > 1e-8

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else float("nan")
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_pct": float(mape),
    }


def train_solar_model():
    data = prepare_solar_dataset()

    target = "target_solar_power_kw"
    features = [column for column in data.columns if column != target]

    # Chronological split: 65% train, 15% validation, 20% test.
    total_samples = len(data)

    train_end = int(total_samples * 0.65)
    validation_end = int(total_samples * 0.80)

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Insufficient data for chronological splitting")

    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

    X_test = test[features]
    y_test = test[target]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    # Validation data is used only for early stopping.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    # Final predictions on unseen test data.
    predictions = model.predict(X_test)

    # Latest observed solar power as persistence baseline.
    baseline_predictions = X_test["solar_lag_1"].to_numpy()

    results = pd.DataFrame({
        "timestamp": test.index,
        "actual_solar_power_kw": y_test.to_numpy(),
        "xgboost_prediction_kw": predictions,
        "baseline_prediction_kw": baseline_predictions,
    })

    # Overall evaluation.
    actual = results["actual_solar_power_kw"].to_numpy()

    overall_xgb = calculate_metrics(actual, predictions)
    overall_baseline = calculate_metrics(
        actual,
        baseline_predictions,
    )

    # Daytime / generating-period evaluation.
    generating_mask = actual > 1e-8
    nighttime_mask = ~generating_mask

    generating_xgb = calculate_metrics(
        actual[generating_mask],
        predictions[generating_mask],
    )

    generating_baseline = calculate_metrics(
        actual[generating_mask],
        baseline_predictions[generating_mask],
    )

    nighttime_xgb = calculate_metrics(
        actual[nighttime_mask],
        predictions[nighttime_mask],
    )

    nighttime_baseline = calculate_metrics(
        actual[nighttime_mask],
        baseline_predictions[nighttime_mask],
    )

    # Save model and metadata.
    artifact = {
        "model": model,
        "features": features,
        "target": target,
        "forecast_horizon_minutes": 15,
        "training_samples": len(train),
        "validation_samples": len(validation),
        "testing_samples": len(test),
        "best_iteration": model.best_iteration,
    }

    model_path = MODEL_DIR / "industrial_solar_xgboost.joblib"
    results_path = MODEL_DIR / "solar_forecast_results.csv"

    joblib.dump(artifact, model_path)
    results.to_csv(results_path, index=False)

    print("\nSOLAR MODEL TRAINING COMPLETED")
    print("=" * 50)
    print("Training samples:", len(train))
    print("Validation samples:", len(validation))
    print("Testing samples:", len(test))
    print("Features:", len(features))
    print("Best iteration:", model.best_iteration)
    print("Model saved:", model_path)
    print("Results saved:", results_path)

    print("\nSOLAR FORECAST EVALUATION")
    print("=" * 50)

    evaluation = {
        "Overall": {
            "XGBoost": overall_xgb,
            "Persistence baseline": overall_baseline,
        },
        "Generating period": {
            "sample_count": int(generating_mask.sum()),
            "XGBoost": generating_xgb,
            "Persistence baseline": generating_baseline,
        },
        "Nighttime": {
            "sample_count": int(nighttime_mask.sum()),
            "XGBoost": nighttime_xgb,
            "Persistence baseline": nighttime_baseline,
        },
    }

    for period, period_metrics in evaluation.items():
        print(f"\n{period}")
        print("-" * 40)

        if "sample_count" in period_metrics:
            print("Samples:", period_metrics["sample_count"])

        for model_name, metrics in period_metrics.items():
            if model_name == "sample_count":
                continue

            print(f"\n{model_name}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    train_solar_model()
