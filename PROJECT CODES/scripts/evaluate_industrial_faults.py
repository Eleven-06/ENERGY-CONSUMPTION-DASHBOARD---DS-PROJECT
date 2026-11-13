
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from bet.analytics.aggregation import load_energy_timeseries
from bet.anomaly.industrial import IndustrialIsolationForest
from bet.anomaly.industrial_rules import IndustrialRuleDetector
from bet.simulation.fault_injection import inject_industrial_faults

OUTPUT_DIR = Path("results/anomaly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RULE_FLAGS = {
    "demand_spike": "rule_demand_spike",
    "demand_drop": "rule_demand_drop",
    "solar_degradation": "rule_solar_degradation",
    "wind_generation_loss": "rule_wind_generation_loss",
    "grid_import_disturbance": "rule_grid_import_disturbance",
    "wind_speed_sensor_stuck": "rule_wind_sensor_flatline",
}


def normalize_timestamps(timestamps):
    """Convert timestamps into a consistent UTC representation."""

    normalized = []

    for timestamp in timestamps:
        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        normalized.append(timestamp)

    return set(normalized)


def calculate_metrics(y_true, y_pred):
    """Calculate binary classification metrics."""

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    ).ravel()

    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
    }


def print_metrics(name, metrics):
    """Display detector evaluation metrics."""

    print(f"\n{name}")
    print("=" * 60)

    print("True positives :", metrics["true_positive"])
    print("False positives:", metrics["false_positive"])
    print("True negatives :", metrics["true_negative"])
    print("False negatives:", metrics["false_negative"])

    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-score            : {metrics['f1_score']:.4f}")
    print(f"Specificity         : {metrics['specificity']:.4f}")
    print(
        f"False positive rate : "
        f"{metrics['false_positive_rate']:.4f}"
    )


def evaluate_predictions(labels, predicted_timestamps):
    """Evaluate timestamp-level anomaly predictions."""

    evaluation = labels.copy()

    evaluation["timestamp"] = pd.to_datetime(
        evaluation["timestamp"],
        utc=True,
    )

    evaluation["predicted_anomaly"] = (
        evaluation["timestamp"].isin(predicted_timestamps)
    )

    metrics = calculate_metrics(
        evaluation["is_fault"],
        evaluation["predicted_anomaly"],
    )

    return evaluation, metrics


def calculate_latency(evaluation, events):
    """Measure first detection latency within each fault interval."""

    rows = []

    for event in events:

        event_start = pd.Timestamp(event.start).tz_convert("UTC")
        event_end = pd.Timestamp(event.end).tz_convert("UTC")

        event_rows = evaluation[
            (evaluation["timestamp"] >= event_start)
            & (evaluation["timestamp"] <= event_end)
        ]

        detections = event_rows[
            event_rows["predicted_anomaly"]
        ]

        if detections.empty:
            detected = False
            latency = None
        else:
            detected = True

            first_detection = detections["timestamp"].min()

            latency = (
                first_detection - event_start
            ).total_seconds() / 60

        rows.append({
            "fault_type": event.fault_type,
            "asset_id": event.asset_id,
            "start": event_start,
            "end": event_end,
            "detected": detected,
            "detection_latency_minutes": latency,
        })

    return pd.DataFrame(rows)


def evaluate_individual_rules(rule_results, labels):
    """
    Evaluate each engineering rule against its matching fault type.

    A rule firing during another fault is not counted as a
    correct classification.
    """

    fault_types = labels["fault_type"].fillna("").to_numpy()
    is_fault = labels["is_fault"].astype(bool).to_numpy()

    rows = []

    print("\nINDIVIDUAL ENGINEERING RULE EVALUATION")
    print("=" * 70)

    for fault_type, flag_column in RULE_FLAGS.items():

        rule_flags = (
            rule_results[flag_column]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        if len(rule_flags) != len(labels):
            raise ValueError(
                f"Length mismatch for {flag_column}: "
                f"{len(rule_flags)} versus {len(labels)}"
            )

        matching_fault = fault_types == fault_type

        # Correctly identified timestamps for this fault.
        true_positive = int(
            (rule_flags & matching_fault).sum()
        )

        # Alerts on timestamps that are not this fault type.
        false_positive = int(
            (rule_flags & ~matching_fault).sum()
        )

        # Target fault timestamps not detected by this rule.
        false_negative = int(
            (~rule_flags & matching_fault).sum()
        )

        clean_alerts = int(
            (rule_flags & ~is_fault).sum()
        )

        total_alerts = int(rule_flags.sum())

        precision = (
            true_positive / total_alerts
            if total_alerts else 0.0
        )

        recall = (
            true_positive / matching_fault.sum()
            if matching_fault.sum() else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        row = {
            "fault_type": fault_type,
            "rule_column": flag_column,
            "ground_truth_count": int(matching_fault.sum()),
            "total_alerts": total_alerts,
            "clean_false_positives": clean_alerts,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        rows.append(row)

        print(f"\nRule: {fault_type}")
        print(f"  Clean-data alerts       : {clean_alerts}")
        print(f"  Total alerts            : {total_alerts}")
        print(f"  Correct detections      : {true_positive}")
        print(f"  Other-type alerts       : {false_positive}")
        print(f"  Missed fault timestamps : {false_negative}")
        print(f"  Precision               : {precision:.4f}")
        print(f"  Recall                  : {recall:.4f}")
        print(f"  F1-score                : {f1:.4f}")

    return pd.DataFrame(rows)


def main():

    # --------------------------------------------------
    # 1. Load original industrial data
    # --------------------------------------------------

    print("\nLOADING INDUSTRIAL ENERGY DATA")
    print("=" * 60)

    data = load_energy_timeseries().sort_index()

    if data.empty:
        raise ValueError("Industrial energy dataset is empty")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex")

    if data.index.has_duplicates:
        raise ValueError("Duplicate timestamps found")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically")

    print("Dataset shape:", data.shape)
    print("Start:", data.index.min())
    print("End:", data.index.max())

    # --------------------------------------------------
    # 2. Chronological split
    # --------------------------------------------------

    split_index = int(len(data) * 0.65)

    baseline = data.iloc[:split_index].copy()
    clean_detection = data.iloc[split_index:].copy()

    print("\nDATA SPLIT")
    print("=" * 60)

    print("Clean baseline samples:", len(baseline))
    print("Detection samples:", len(clean_detection))

    # --------------------------------------------------
    # 3. Inject controlled faults
    # --------------------------------------------------

    injected_data, labels, events = inject_industrial_faults(
        clean_detection,
        duration_intervals=4,
    )

    labels = labels.reset_index(drop=True)

    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"],
        utc=True,
    )

    if len(injected_data) != len(labels):
        raise ValueError("Injected data and ground truth lengths differ")

    if not injected_data.index.equals(
        pd.DatetimeIndex(labels["timestamp"])
    ):
        raise ValueError(
            "Injected data timestamps do not match ground-truth timestamps"
        )

    print("\nINJECTED FAULT DETAILS")
    print("=" * 60)

    for event in events:
        print(
            f"{event.fault_type:30s} "
            f"| Asset: {event.asset_id:15s} "
            f"| Start: {event.start} "
            f"| End: {event.end}"
        )

    # --------------------------------------------------
    # 4. Save injected data and labels
    # --------------------------------------------------

    injected_export = injected_data.copy()
    injected_export.index.name = "timestamp"

    injected_export.to_csv(
        OUTPUT_DIR / "fault_injected_timeseries.csv"
    )

    labels.to_csv(
        OUTPUT_DIR / "fault_ground_truth.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Isolation Forest
    # --------------------------------------------------

    print("\nTRAINING ISOLATION FOREST")
    print("=" * 60)

    isolation_detector = IndustrialIsolationForest()
    isolation_detector.fit(baseline)

    print("Isolation Forest trained successfully.")

    isolation_results = isolation_detector.detect(
        injected_data
    )

    isolation_timestamps = normalize_timestamps(
        result.timestamp for result in isolation_results
    )

    isolation_evaluation, isolation_metrics = evaluate_predictions(
        labels,
        isolation_timestamps,
    )

    print_metrics(
        "ISOLATION FOREST RESULTS",
        isolation_metrics,
    )

    isolation_latency = calculate_latency(
        isolation_evaluation,
        events,
    )

    # --------------------------------------------------
    # 6. Engineering-rule detector
    # --------------------------------------------------

    print("\nRUNNING ENGINEERING-RULE DETECTOR")
    print("=" * 60)

    rule_detector = IndustrialRuleDetector()

    # Run on clean and injected data separately.
    clean_rule_results = rule_detector.detect(clean_detection)

    rule_results = rule_detector.detect(injected_data)

    missing_columns = [
        column
        for column in RULE_FLAGS.values()
        if column not in rule_results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing engineering-rule columns: {missing_columns}"
        )

    # Verify rule-result timestamps match injected data.
    if not rule_results.index.equals(injected_data.index):
        raise ValueError(
            "Engineering-rule output timestamps do not match input"
        )

    # --------------------------------------------------
    # 7. Corrected individual rule diagnostics
    # --------------------------------------------------

    individual_rule_metrics = evaluate_individual_rules(
        rule_results.reset_index(drop=True),
        labels,
    )

    individual_rule_metrics.to_csv(
        OUTPUT_DIR / "individual_rule_results.csv",
        index=False,
    )

    # --------------------------------------------------
    # 8. Combined engineering-rule evaluation
    # --------------------------------------------------

    rule_results["predicted_anomaly"] = (
        rule_results[list(RULE_FLAGS.values())]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )

    rule_timestamps = normalize_timestamps(
        rule_results.index[
            rule_results["predicted_anomaly"]
        ]
    )

    rule_evaluation, rule_metrics = evaluate_predictions(
        labels,
        rule_timestamps,
    )

    print_metrics(
        "ENGINEERING-RULE DETECTOR RESULTS",
        rule_metrics,
    )

    rule_latency = calculate_latency(
        rule_evaluation,
        events,
    )

    # --------------------------------------------------
    # 9. Clean-data false-positive analysis
    # --------------------------------------------------

    print("\nENGINEERING RULE FALSE-POSITIVE ANALYSIS")
    print("=" * 70)

    clean_rule_counts = {}

    for fault_type, flag_column in RULE_FLAGS.items():

        clean_alerts = int(
            clean_rule_results[flag_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        clean_rule_counts[fault_type] = clean_alerts

        print(
            f"{fault_type:30s}: "
            f"{clean_alerts} alerts on clean data"
        )

    # --------------------------------------------------
    # 10. Per-fault combined detection summary
    # --------------------------------------------------

    print("\nPER-FAULT COMBINED DETECTION SUMMARY")
    print("=" * 60)

    for event in events:

        event_rows = rule_evaluation[
            rule_evaluation["fault_type"] == event.fault_type
        ]

        detected = int(
            event_rows["predicted_anomaly"].sum()
        )

        total = len(event_rows)

        print(
            f"{event.fault_type:30s}: "
            f"{detected}/{total} detected"
        )

    # --------------------------------------------------
    # 11. Detector comparison
    # --------------------------------------------------

    comparison = pd.DataFrame([
        {
            "detector": "Isolation Forest",
            **isolation_metrics,
        },
        {
            "detector": "Engineering Rules",
            **rule_metrics,
        },
    ])

    print("\nDETECTOR COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "detector",
                "precision",
                "recall",
                "f1_score",
                "specificity",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 12. Save evaluation results
    # --------------------------------------------------

    isolation_evaluation.to_csv(
        OUTPUT_DIR / "isolation_forest_evaluation.csv",
        index=False,
    )

    rule_evaluation.to_csv(
        OUTPUT_DIR / "engineering_rule_evaluation.csv",
        index=False,
    )

    rule_results.to_csv(
        OUTPUT_DIR / "engineering_rule_results.csv",
        index_label="timestamp",
    )

    isolation_latency.to_csv(
        OUTPUT_DIR / "isolation_forest_latency.csv",
        index=False,
    )

    rule_latency.to_csv(
        OUTPUT_DIR / "engineering_rule_latency.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "detector_comparison.csv",
        index=False,
    )

    print("\nSAVED OUTPUT FILES")
    print("=" * 60)

    output_files = [
        "fault_injected_timeseries.csv",
        "fault_ground_truth.csv",
        "isolation_forest_evaluation.csv",
        "engineering_rule_evaluation.csv",
        "engineering_rule_results.csv",
        "isolation_forest_latency.csv",
        "engineering_rule_latency.csv",
        "detector_comparison.csv",
        "individual_rule_results.csv",
    ]

    for filename in output_files:
        print(OUTPUT_DIR / filename)

    print("\nIndustrial fault evaluation completed.")


if __name__ == "__main__":
    main()
