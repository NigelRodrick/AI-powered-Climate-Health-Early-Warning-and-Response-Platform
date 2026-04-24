from __future__ import annotations

import pandas as pd


WEIGHTS = {
    "flood_risk_index": 0.30,
    "clinic_case_load_index": 0.30,
    "medicine_stock_gap_index": 0.20,
    "temperature_stress_index": 0.20,
}


def _temperature_stress_index(temperature_c: float) -> float:
    """
    Convert temperature to a simple stress score.
    28C and below is treated as low stress, 42C+ as max stress.
    """
    low, high = 28.0, 42.0
    normalized = (temperature_c - low) / (high - low)
    return float(max(0.0, min(1.0, normalized)))


def classify_risk(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def score_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with risk_score and risk_level columns."""
    scored = df.copy()
    scored["temperature_stress_index"] = scored["temperature_c"].apply(_temperature_stress_index)
    scored["contrib_flood"] = scored["flood_risk_index"] * WEIGHTS["flood_risk_index"]
    scored["contrib_case_load"] = scored["clinic_case_load_index"] * WEIGHTS["clinic_case_load_index"]
    scored["contrib_stock_gap"] = scored["medicine_stock_gap_index"] * WEIGHTS["medicine_stock_gap_index"]
    scored["contrib_temperature"] = scored["temperature_stress_index"] * WEIGHTS["temperature_stress_index"]

    weighted = scored["contrib_flood"] + scored["contrib_case_load"] + scored["contrib_stock_gap"] + scored["contrib_temperature"]

    scored["risk_score"] = weighted.round(3)
    scored["risk_level"] = scored["risk_score"].apply(classify_risk)
    return scored
