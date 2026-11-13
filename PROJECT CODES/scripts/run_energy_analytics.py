
from bet.analytics import (
    calculate_daily_energy,
    calculate_energy_kpis,
    load_energy_timeseries,
)


def main():
    print("Loading industrial telemetry...")
    data = load_energy_timeseries()

    print("\nTimeseries shape:", data.shape)

    print("\nINDUSTRIAL ENERGY KPIs")
    print("=" * 45)

    kpis = calculate_energy_kpis(data)

    for name, value in kpis.items():
        print(f"{name:40s}: {value:,.3f}")

    print("\nDAILY ENERGY SUMMARY")
    print("=" * 45)

    daily = calculate_daily_energy(data)
    print(daily.head(10).to_string())

    daily.to_csv(
        "data/simulated/daily_energy_summary.csv"
    )

    print("\nDaily summary saved successfully.")


if __name__ == "__main__":
    main()
