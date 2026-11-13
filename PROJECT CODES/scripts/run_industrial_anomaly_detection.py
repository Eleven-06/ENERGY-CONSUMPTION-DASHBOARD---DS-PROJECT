
from pathlib import Path

import pandas as pd

from bet.analytics.aggregation import load_energy_timeseries
from bet.anomaly.industrial import IndustrialIsolationForest

OUTPUT_DIR = Path("results/anomaly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = load_energy_timeseries().sort_index()

    # Chronological baseline split.
    split_index = int(len(data) * 0.65)

    baseline = data.iloc[:split_index]
    detection_data = data.iloc[split_index:]

    detector = IndustrialIsolationForest(
        contamination=0.01,
        random_state=42,
    )

    detector.fit(baseline)

    anomalies = detector.detect(detection_data)

    records = [
        anomaly.to_dict()
        for anomaly in anomalies
    ]

    output_path = OUTPUT_DIR / "industrial_anomalies.csv"

    if records:
        results = pd.DataFrame(records)
    else:
        results = pd.DataFrame(
            columns=[
                "timestamp",
                "severity",
                "anomaly_score",
                "measurements",
                "explanation",
            ]
        )

    results.to_csv(output_path, index=False)

    print("\nINDUSTRIAL ANOMALY DETECTION")
    print("=" * 50)
    print("Baseline samples:", len(baseline))
    print("Detection samples:", len(detection_data))
    print("Detected anomalies:", len(anomalies))
    print("Output:", output_path)

    if len(results):
        print("\nSeverity summary:")
        print(results["severity"].value_counts().to_string())

        print("\nSample anomalies:")
        print(
            results[
                ["timestamp", "severity", "anomaly_score", "explanation"]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()
