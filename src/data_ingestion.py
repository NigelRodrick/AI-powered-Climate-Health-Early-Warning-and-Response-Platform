from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "location_id",
    "week",
    "rainfall_mm",
    "temperature_c",
    "flood_risk_index",
    "clinic_case_load_index",
    "medicine_stock_gap_index",
}


def load_input_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate input data for risk scoring."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp bounded indexes to expected ranges and return a copy."""
    out = df.copy()
    bounded = ["flood_risk_index", "clinic_case_load_index", "medicine_stock_gap_index"]
    for col in bounded:
        out[col] = out[col].clip(lower=0.0, upper=1.0)
    out["rainfall_mm"] = out["rainfall_mm"].clip(lower=0.0)
    return out
