
from bet.industrial_forecasting.dataset import (
    chronological_split,
    prepare_demand_dataset,
)


def main():
    data = prepare_demand_dataset()
    train, test = chronological_split(data)

    target = "target_demand_kw"
    features = [column for column in data.columns if column != target]

    print("NEXT-INTERVAL DEMAND FORECASTING DATASET")
    print("=" * 50)

    print("Total samples:", len(data))
    print("Training samples:", len(train))
    print("Testing samples:", len(test))

    print("\nTarget:", target)
    print("Forecast horizon: 15 minutes")
    print("Feature count:", len(features))

    print("\nTraining period:")
    print(train.index.min(), "to", train.index.max())

    print("\nTesting period:")
    print(test.index.min(), "to", test.index.max())

    print("\nFeatures:")
    print(features)

    assert target not in features

    data.to_csv(
        "data/simulated/demand_forecasting_dataset.csv"
    )

    print("\nDataset saved successfully.")


if __name__ == "__main__":
    main()
