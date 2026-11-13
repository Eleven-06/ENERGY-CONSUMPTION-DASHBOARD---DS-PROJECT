
from pathlib import Path

import pandas as pd

from bet.industrial.schema import EnergyReading

REQUIRED_COLUMNS = {
    "timestamp_utc",
    "facility_id",
    "asset_id",
    "measurement_type",
    "value",
    "unit",
    "source_type",
    "quality_flag",
    "dataset_version",
}


def ingest_industrial_csv(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    valid_records = []
    rejected_records = []

    for index, row in raw.iterrows():
        try:
            record = row.to_dict()

            # Normalize dataset version to string
            record["dataset_version"] = str(record["dataset_version"])

            reading = EnergyReading.from_dict(record)
            valid_records.append(reading.to_dict())
        except (ValueError, TypeError, KeyError) as exc:
            rejected_records.append({
                "row": int(index),
                "reason": str(exc),
            })

    clean = pd.DataFrame(valid_records)

    if not clean.empty:
        clean["timestamp_utc"] = pd.to_datetime(
            clean["timestamp_utc"], utc=True
        )

        duplicate_count = int(
            clean.duplicated(
                subset=[
                    "timestamp_utc",
                    "facility_id",
                    "asset_id",
                    "measurement_type",
                ]
            ).sum()
        )

        clean = clean.drop_duplicates(
            subset=[
                "timestamp_utc",
                "facility_id",
                "asset_id",
                "measurement_type",
            ],
            keep="first",
        )

        clean = clean.sort_values(
            ["timestamp_utc", "facility_id", "asset_id"]
        ).reset_index(drop=True)
    else:
        duplicate_count = 0

    report = {
        "input_records": len(raw),
        "valid_records_before_deduplication": len(valid_records),
        "rejected_records": len(rejected_records),
        "duplicate_records_removed": duplicate_count,
        "output_records": len(clean),
    }

    return clean, report, rejected_records
