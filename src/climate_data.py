from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from service import LOCATION_COORDS


OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_climate_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily climate history for Zimbabwe reference districts from Open-Meteo.
    Returns one row per district-day.
    """
    rows: list[dict[str, Any]] = []
    for location_id, meta in LOCATION_COORDS.items():
        params = {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Africa/Harare",
        }
        response = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json().get("daily", {})
        times = payload.get("time", [])
        temps = payload.get("temperature_2m_mean", [])
        rain = payload.get("precipitation_sum", [])
        for t, temp, pr in zip(times, temps, rain):
            rows.append(
                {
                    "location_id": location_id,
                    "district": meta["district"],
                    "date": t,
                    "temperature_c": temp,
                    "rainfall_mm": pr,
                }
            )
    return pd.DataFrame(rows)


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week"] = out["date"].dt.strftime("%Y-W%U")
    weekly = (
        out.groupby(["location_id", "district", "week"], as_index=False)
        .agg(temperature_c=("temperature_c", "mean"), rainfall_mm=("rainfall_mm", "sum"))
    )
    # Derived climate stress features for model training.
    weekly["temp_anomaly"] = weekly["temperature_c"] - weekly["temperature_c"].mean()
    weekly["rain_anomaly"] = weekly["rainfall_mm"] - weekly["rainfall_mm"].mean()
    weekly["flood_risk_index"] = (weekly["rainfall_mm"] / (weekly["rainfall_mm"].max() + 1e-6)).clip(0, 1)
    return weekly


def default_date_range() -> tuple[str, str]:
    end = date.today()
    start = date(end.year - 2, end.month, max(1, end.day))
    return start.isoformat(), end.isoformat()
