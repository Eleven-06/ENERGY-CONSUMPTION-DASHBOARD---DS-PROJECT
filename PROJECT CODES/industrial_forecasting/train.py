
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from bet.industrial_forecasting.dataset import prepare_demand_dataset
from bet.industrial_forecasting.evaluate import evaluate_forecasts

MODEL_DIR = Path("models/industrial")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_demand_model():
    data = prepare_demand_dataset()

    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    target = "target_demand_kw"
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

    # Early stopping uses validation data, never the final test data.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    # Final evaluation on previously unseen test data.
    predictions = model.predict(X_test)

    # Persistence baseline: previous interval's observed demand.
    baseline_predictions = X_test["lag_1"].to_numpy()

    results = pd.DataFrame({
        "timestamp": test.index,
        "actual_demand_kw": y_test.to_numpy(),
        "xgboost_prediction_kw": predictions,
        "baseline_prediction_kw": baseline_predictions,
    })

    metrics = evaluate_forecasts(results)

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

    model_path = MODEL_DIR / "industrial_demand_xgboost.joblib"
    results_path = MODEL_DIR / "demand_forecast_results.csv"

    joblib.dump(artifact, model_path)
    results.to_csv(results_path, index=False)

    print("\nMODEL TRAINING COMPLETED")
    print("=" * 50)
    print("Training samples:", len(train))
    print("Validation samples:", len(validation))
    print("Testing samples:", len(test))
    print("Features:", len(features))
    print("Best iteration:", model.best_iteration)
    print("Model saved:", model_path)
    print("Results saved:", results_path)

    print("\nFORECAST MODEL EVALUATION")
    print("=" * 50)

    for period, period_metrics in metrics.items():
        print(f"\n{period}")
        print("-" * 40)

        if "sample_count" in period_metrics:
            print("Samples:", period_metrics["sample_count"])

        for model_name, model_metrics in period_metrics.items():
            if model_name == "sample_count":
                continue

            print(f"\n{model_name}")

            for metric_name, value in model_metrics.items():
                print(f"{metric_name}: {value:.4f}")

    return results


if __name__ == "__main__":
    train_demand_model()
