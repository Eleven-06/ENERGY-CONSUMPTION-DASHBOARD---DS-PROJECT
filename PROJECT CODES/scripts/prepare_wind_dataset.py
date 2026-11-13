
from pathlib import Path

from bet.industrial_forecasting.wind_dataset import (
    prepare_wind_dataset,
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = prepare_wind_dataset()

    output_path = OUTPUT_DIR / "wind_forecasting_dataset.csv"

    data.to_csv(output_path, index_label="timestamp")

    print("\nWIND DATASET PREPARATION")
    print("=" * 45)
    print("Total samples:", len(data))
    print("Features:", len(data.columns) - 1)
    print("Target: target_wind_power_kw")
    print("Forecast horizon: 15 minutes")
    print("Dataset saved:", output_path)

    print("\nWind target statistics:")
    print(data["target_wind_power_kw"].describe())


if __name__ == "__main__":
    main()
