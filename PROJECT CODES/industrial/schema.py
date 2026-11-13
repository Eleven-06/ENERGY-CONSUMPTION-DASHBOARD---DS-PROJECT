
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from typing import Literal

SourceType = Literal["public", "simulated", "iot"]
QualityFlag = Literal["good", "estimated", "suspect"]

MEASUREMENT_UNITS = {
    "power_kw": "kW",
    "energy_kwh": "kWh",
    "irradiance_w_m2": "W/m2",
    "wind_speed_m_s": "m/s",
    "battery_soc_pct": "%",
    "temperature_c": "degC",
    "fuel_volume_l": "L",
    "gas_volume_m3": "m3",
}

VALID_SOURCES = {"public", "simulated", "iot"}
VALID_QUALITY_FLAGS = {"good", "estimated", "suspect"}


@dataclass(frozen=True)
class EnergyReading:
    timestamp_utc: datetime
    facility_id: str
    asset_id: str | None
    measurement_type: str
    value: float
    unit: str
    source_type: SourceType
    quality_flag: QualityFlag = "good"
    dataset_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_utc, datetime):
            raise TypeError("timestamp_utc must be a datetime")

        if (
            self.timestamp_utc.tzinfo is None
            or self.timestamp_utc.utcoffset() is None
        ):
            raise ValueError("timestamp_utc must be timezone-aware")

        if not self.facility_id.strip():
            raise ValueError("facility_id cannot be empty")

        if self.asset_id is not None and not self.asset_id.strip():
            raise ValueError("asset_id cannot be empty")

        if self.measurement_type not in MEASUREMENT_UNITS:
            raise ValueError(
                f"Unsupported measurement_type: {self.measurement_type}"
            )

        expected_unit = MEASUREMENT_UNITS[self.measurement_type]

        if self.unit != expected_unit:
            raise ValueError(
                f"{self.measurement_type} requires unit {expected_unit}"
            )

        if not isinstance(self.value, (int, float)) or isinstance(
            self.value, bool
        ):
            raise TypeError("value must be numeric")

        if not isfinite(self.value):
            raise ValueError("value must be finite")

        if self.measurement_type != "temperature_c" and self.value < 0:
            raise ValueError(
                f"{self.measurement_type} cannot be negative"
            )

        if self.measurement_type == "battery_soc_pct":
            if not 0 <= self.value <= 100:
                raise ValueError("battery_soc_pct must be between 0 and 100")

        if self.source_type not in VALID_SOURCES:
            raise ValueError(f"Invalid source_type: {self.source_type}")

        if self.quality_flag not in VALID_QUALITY_FLAGS:
            raise ValueError(f"Invalid quality_flag: {self.quality_flag}")

        if not self.dataset_version.strip():
            raise ValueError("dataset_version cannot be empty")

    def to_dict(self) -> dict:
        record = asdict(self)
        record["timestamp_utc"] = self.timestamp_utc.isoformat()
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "EnergyReading":
        data = dict(record)
        data["timestamp_utc"] = datetime.fromisoformat(
            data["timestamp_utc"]
        )
        return cls(**data)
