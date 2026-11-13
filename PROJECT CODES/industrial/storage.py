
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from bet.database.connection import engine
from bet.industrial.ingestion import ingest_industrial_csv

TABLE_NAME = "industrial_energy_readings"


def load_industrial_csv(path: str | Path):
    # Validate and clean the source dataset
    data, report, rejected = ingest_industrial_csv(path)

    if rejected:
        raise ValueError(
            f"Dataset contains {len(rejected)} rejected records"
        )

    if data.empty:
        raise ValueError("No valid records to insert")

    # Normalize database column types
    data["timestamp_utc"] = pd.to_datetime(
        data["timestamp_utc"], utc=True
    )
    data["dataset_version"] = data["dataset_version"].astype(str)

    # Use one transaction for the count check and insertion
    with engine.begin() as connection:
        existing_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        ).scalar_one()

        if existing_count > 0:
            raise RuntimeError(
                f"{TABLE_NAME} already contains {existing_count} rows. "
                "Refusing to insert duplicate dataset."
            )

        inserted = data.to_sql(
            TABLE_NAME,
            con=connection,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )

    return {
        "inserted_records": inserted,
        "ingestion_report": report,
    }
