
from bet.industrial.storage import load_industrial_csv


def main():
    dataset = "data/simulated/industrial_factory_15min.csv"

    print("Loading industrial telemetry into PostgreSQL...")

    result = load_industrial_csv(dataset)

    print("\nDatabase insertion successful!")
    print("Inserted records:", result["inserted_records"])
    print("Ingestion report:", result["ingestion_report"])


if __name__ == "__main__":
    main()
