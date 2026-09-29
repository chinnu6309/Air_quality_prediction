"""Create a synthetic, air-quality-like CSV so the project can be run end-to-end.

This file is for demonstration only. Use a trusted measured dataset for your final
submission and clearly cite its source in your report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    rows = 730  # two daily readings over one year
    dates = pd.date_range("2024-01-01", periods=rows, freq="12h")

    seasonal = 18 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365)
    pm25 = np.clip(58 + seasonal + rng.normal(0, 15, rows), 3, None)
    pm10 = np.clip(pm25 * rng.normal(1.45, 0.15, rows) + rng.normal(0, 8, rows), 5, None)
    no2 = np.clip(28 + 0.20 * pm25 + rng.normal(0, 7, rows), 2, None)
    so2 = np.clip(7 + rng.normal(0, 2.5, rows), 0.5, None)
    co = np.clip(0.45 + 0.008 * pm25 + rng.normal(0, 0.12, rows), 0.05, None)
    o3 = np.clip(58 - 0.18 * pm25 + rng.normal(0, 9, rows), 4, None)
    temperature = 28 + 7 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 70) / 365) + rng.normal(0, 2, rows)
    humidity = np.clip(64 - 0.28 * (temperature - 25) + rng.normal(0, 9, rows), 20, 98)
    wind_speed = np.clip(rng.gamma(2.2, 1.2, rows), 0.1, 15)

    # A synthetic continuous target designed solely to demonstrate regression.
    aqi = (
        0.72 * pm25
        + 0.18 * pm10
        + 0.35 * no2
        + 11 * co
        - 0.12 * o3
        + 0.08 * humidity
        - 1.4 * wind_speed
        + rng.normal(0, 11, rows)
    )

    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "Temperature": temperature,
            "Humidity": humidity,
            "WindSpeed": wind_speed,
            "AQI": np.clip(aqi, 1, 500),
        }
    )

    # Inject a few missing feature values so preprocessing can be demonstrated.
    for column in ("PM2.5", "NO2", "Humidity"):
        frame.loc[rng.choice(rows, size=8, replace=False), column] = np.nan

    output = Path(__file__).parent / "data" / "air_quality.csv"
    frame.to_csv(output, index=False)
    print(f"Demo data written to: {output}")
    print("Rows:", len(frame))


if __name__ == "__main__":
    main()
