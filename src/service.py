from __future__ import annotations

from pathlib import Path
from typing import Any
import io
import json

import joblib
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from alerting import build_alerts
from data_ingestion import normalize_features
from risk_scoring import score_locations

ROOT = Path(__file__).resolve().parent.parent
ACTION_STATUS_PATH = ROOT / "sample_data" / "action_status.json"
MODEL_PATH = ROOT / "models" / "outbreak_model.joblib"

LOCATION_COORDS = {
    "ZW-Harare-001": {"district": "Harare", "lat": -17.8292, "lon": 31.0522},
    "ZW-Bulawayo-002": {"district": "Bulawayo", "lat": -20.1325, "lon": 28.6265},
    "ZW-Mutare-003": {"district": "Mutare", "lat": -18.9707, "lon": 32.6709},
    "ZW-Masvingo-004": {"district": "Masvingo", "lat": -20.0744, "lon": 30.8327},
    "ZW-Gweru-005": {"district": "Gweru", "lat": -19.455, "lon": 29.8167},
}


def score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score raw records and return alert payloads."""
    if not records:
        return []

    df = pd.DataFrame(records)
    df = normalize_features(df)
    scored = score_locations(df)
    scored = apply_ml_inference(scored)
    return enrich_alerts(build_alerts(scored), scored)


def score_sample_file(csv_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    return score_records(df.to_dict(orient="records"))


def _driver_text(row: pd.Series) -> str:
    drivers = {
        "Flood": float(row["contrib_flood"]),
        "CaseLoad": float(row["contrib_case_load"]),
        "StockGap": float(row["contrib_stock_gap"]),
        "Temperature": float(row["contrib_temperature"]),
    }
    top = sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:2]
    return ", ".join(f"{k}:{v:.2f}" for k, v in top)


def enrich_alerts(alerts: list[dict[str, Any]], scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    mean_score = float(scored_df["risk_score"].mean())
    std_score = float(scored_df["risk_score"].std(ddof=0) or 0.0)

    status_map = read_action_status()

    for i, alert in enumerate(alerts):
        row = scored_df.iloc[i]
        score = float(row["risk_score"])
        z = (score - mean_score) / std_score if std_score > 0 else 0.0
        alert["ai_top_drivers"] = _driver_text(row)
        alert["anomaly_flag"] = z > 1.0
        alert["model_version"] = row.get("model_version", "risk-v0.2")
        alert["outbreak_probability"] = round(float(row.get("outbreak_probability", 0.0)), 3)
        alert["action_status"] = status_map.get(alert["location_id"], "new")
        alert.update(LOCATION_COORDS.get(alert["location_id"], {}))
    return alerts


def apply_ml_inference(scored_df: pd.DataFrame) -> pd.DataFrame:
    out = scored_df.copy()
    if not MODEL_PATH.exists():
        out["outbreak_probability"] = out["risk_score"]
        out["model_version"] = "heuristic-v0.2"
        return out

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_columns = artifact.get("feature_columns", [])
        source = artifact.get("training_source", "trained-model")
    else:
        model = artifact
        feature_columns = []
        source = "trained-model"

    # Defensive unwrapping for legacy nested artifact structures.
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    base_features = pd.DataFrame(
        {
            "Temperature (C)": out["temperature_c"],
            "Humidity": (out["rainfall_mm"] / 200.0).clip(0.0, 1.0),
            "Wind Speed (km/h)": (7.0 + out["flood_risk_index"] * 12.0).clip(0.0, 45.0),
            "Age": 30,
            "Gender": 0,
            "temperature_c": out["temperature_c"],
            "rainfall_mm": out["rainfall_mm"],
            "temp_anomaly": out["temperature_c"] - out["temperature_c"].mean(),
            "rain_anomaly": out["rainfall_mm"] - out["rainfall_mm"].mean(),
            "flood_risk_index": out["flood_risk_index"],
        }
    )

    if feature_columns:
        for col in feature_columns:
            if col not in base_features.columns:
                base_features[col] = 0.0
        features = base_features[feature_columns]
    else:
        features = base_features

    probs = model.predict_proba(features)[:, 1]
    out["outbreak_probability"] = probs
    out["risk_score"] = (0.55 * out["risk_score"] + 0.45 * out["outbreak_probability"]).round(3)
    out["risk_level"] = out["risk_score"].apply(
        lambda s: "critical" if s >= 0.75 else "high" if s >= 0.55 else "moderate" if s >= 0.35 else "low"
    )
    out["model_version"] = f"logreg-{source}-v1"
    return out


def filter_by_audience(alerts: list[dict[str, Any]], audience: str) -> list[dict[str, Any]]:
    if not audience:
        return alerts
    filtered: list[dict[str, Any]] = []
    for alert in alerts:
        actions = [a for a in alert["actions"] if a["audience"] == audience]
        if actions:
            copy_alert = dict(alert)
            copy_alert["actions"] = actions
            filtered.append(copy_alert)
    return filtered


def apply_simulation(records: list[dict[str, Any]], rainfall_factor: float, temp_offset: float) -> list[dict[str, Any]]:
    simulated = []
    for r in records:
        item = dict(r)
        item["rainfall_mm"] = float(item["rainfall_mm"]) * rainfall_factor
        item["temperature_c"] = float(item["temperature_c"]) + temp_offset
        simulated.append(item)
    return simulated


def build_trend_series(alerts: list[dict[str, Any]]) -> dict[str, list[float]]:
    trend = {}
    for a in alerts:
        base = float(a["risk_score"])
        trend[a["location_id"]] = [
            max(0.0, round(base - 0.12, 3)),
            max(0.0, round(base - 0.08, 3)),
            max(0.0, round(base - 0.04, 3)),
            round(base, 3),
        ]
    return trend


def read_action_status() -> dict[str, str]:
    if not ACTION_STATUS_PATH.exists():
        return {}
    return json.loads(ACTION_STATUS_PATH.read_text(encoding="utf-8"))


def update_action_status(location_id: str, status: str) -> None:
    state = read_action_status()
    state[location_id] = status
    ACTION_STATUS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def export_csv(alerts: list[dict[str, Any]]) -> str:
    rows = []
    for a in alerts:
        rows.append(
            {
                "location_id": a["location_id"],
                "district": a.get("district", ""),
                "week": a["week"],
                "risk_score": a["risk_score"],
                "outbreak_probability": a.get("outbreak_probability", 0.0),
                "risk_level": a["risk_level"],
                "ai_top_drivers": a["ai_top_drivers"],
                "anomaly_flag": a["anomaly_flag"],
                "action_status": a["action_status"],
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def export_pdf(alerts: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Climate-Health Dashboard Snapshot", styles["Title"]), Spacer(1, 8)]
    for a in alerts:
        text = (
            f"{a['location_id']} ({a.get('district', 'N/A')}) - {a['risk_level'].upper()} | "
            f"score {a['risk_score']} | drivers {a['ai_top_drivers']} | status {a['action_status']}"
        )
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 4))
    doc.build(story)
    return buffer.getvalue()
