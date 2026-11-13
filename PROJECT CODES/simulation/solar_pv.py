
def solar_pv_power(
    irradiance_w_m2: float,
    ambient_temp_c: float,
    capacity_kw: float,
    temperature_coefficient: float = -0.004,
    noct_c: float = 45.0,
) -> float:
    """Estimate solar PV output using irradiance and cell temperature."""

    if capacity_kw <= 0:
        raise ValueError("capacity_kw must be positive")

    if irradiance_w_m2 < 0:
        raise ValueError("irradiance cannot be negative")

    if noct_c <= 20:
        raise ValueError("noct_c must be greater than 20")

    cell_temp_c = ambient_temp_c + (
        (noct_c - 20.0) / 800.0
    ) * irradiance_w_m2

    temperature_factor = max(
        0.0,
        1.0 + temperature_coefficient * (cell_temp_c - 25.0),
    )

    output_kw = (
        capacity_kw
        * irradiance_w_m2 / 1000.0
        * temperature_factor
    )

    return round(min(output_kw, capacity_kw), 3)
