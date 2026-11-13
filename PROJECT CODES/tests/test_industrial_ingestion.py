

from bet.industrial.ingestion import ingest_industrial_csv

DATASET = "data/simulated/industrial_factory_15min.csv"


def test_ingestion_loads_all_records():
    data, report, rejected = ingest_industrial_csv(DATASET)

    assert report["input_records"] == 86400
    assert report["output_records"] == 86400
    assert len(rejected) == 0


def test_ingestion_has_no_duplicates():
    data, report, rejected = ingest_industrial_csv(DATASET)

    assert report["duplicate_records_removed"] == 0


def test_ingestion_timestamps_are_timezone_aware():
    data, report, rejected = ingest_industrial_csv(DATASET)

    assert str(data["timestamp_utc"].dt.tz) == "UTC"


def test_ingestion_contains_required_columns():
    data, report, rejected = ingest_industrial_csv(DATASET)

    required = {
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

    assert required.issubset(data.columns)


def test_ingestion_output_is_time_sorted():
    data, report, rejected = ingest_industrial_csv(DATASET)

    assert data["timestamp_utc"].is_monotonic_increasing
