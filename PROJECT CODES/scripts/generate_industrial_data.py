
from pathlib import Path

from bet.simulation import simulate_factory


def main():
    output = Path("data/simulated/industrial_factory_15min.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Generating industrial energy dataset...")

    data = simulate_factory(
        start="2026-01-01T00:00:00+00:00",
        periods=96 * 30,
        interval_minutes=15,
        seed=42,
    )

    data.to_csv(output, index=False)

    print("\nDataset generated successfully!")
    print(f"Records: {len(data):,}")
    print(f"Unique timestamps: {data['timestamp_utc'].nunique():,}")
    print(f"Unique assets: {data['asset_id'].nunique()}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
