
def wind_turbine_power(
    wind_speed_m_s: float,
    rated_power_kw: float,
    cut_in_m_s: float = 3.0,
    rated_speed_m_s: float = 12.0,
    cut_out_m_s: float = 25.0,
) -> float:
    """Estimate wind turbine power using a simplified power curve."""

    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive")

    if not 0 <= cut_in_m_s < rated_speed_m_s < cut_out_m_s:
        raise ValueError("Invalid turbine operating speeds")

    if wind_speed_m_s < 0:
        raise ValueError("Wind speed cannot be negative")

    if wind_speed_m_s < cut_in_m_s:
        return 0.0

    if wind_speed_m_s >= cut_out_m_s:
        return 0.0

    if wind_speed_m_s >= rated_speed_m_s:
        return round(rated_power_kw, 3)

    normalized = (
        wind_speed_m_s**3 - cut_in_m_s**3
    ) / (
        rated_speed_m_s**3 - cut_in_m_s**3
    )

    return round(rated_power_kw * normalized, 3)
