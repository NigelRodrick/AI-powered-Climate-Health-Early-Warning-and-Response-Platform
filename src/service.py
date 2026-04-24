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
KAGGLE_PATH = ROOT / "sample_data" / "kaggle_disease_dataset.csv"

LOCATION_COORDS = {
    "ZW-Harare-001": {"district": "Harare", "city": "Harare", "region": "Harare Metropolitan", "lat": -17.8292, "lon": 31.0522},
    "ZW-Bulawayo-002": {"district": "Bulawayo", "city": "Bulawayo", "region": "Bulawayo Metropolitan", "lat": -20.1325, "lon": 28.6265},
    "ZW-Mutare-003": {"district": "Mutare", "city": "Mutare", "region": "Manicaland", "lat": -18.9707, "lon": 32.6709},
    "ZW-Masvingo-004": {"district": "Masvingo", "city": "Masvingo", "region": "Masvingo", "lat": -20.0744, "lon": 30.8327},
    "ZW-Gweru-005": {"district": "Gweru", "city": "Gweru", "region": "Midlands", "lat": -19.455, "lon": 29.8167},
}

DISEASE_TYPE_KEYWORDS = {
    "Infectious": ["flu", "influenza", "cold", "malaria", "dengue", "cholera", "covid", "infection", "pneumonia"],
    "Cardiovascular": ["heart", "hypertension", "stroke", "cardio"],
    "Respiratory": ["asthma", "sinus", "respiratory", "bronch"],
    "Gastrointestinal": ["diarrhea", "stomach", "abdominal", "gastro", "vomit"],
    "Neurological": ["migraine", "headache", "neuro", "confusion"],
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


def export_daily_brief(alerts: list[dict[str, Any]], messages: list[dict[str, str]]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Daily Climate-Health Risk Brief", styles["Title"]), Spacer(1, 8)]

    top_alerts = sorted(alerts, key=lambda a: float(a.get("risk_score", 0.0)), reverse=True)[:3]
    story.append(Paragraph("Top Priority Alerts", styles["Heading2"]))
    for a in top_alerts:
        story.append(
            Paragraph(
                f"{a.get('city','N/A')} ({a.get('region','N/A')}): {a.get('risk_level','N/A').upper()} | "
                f"score {a.get('risk_score','N/A')} | outbreak prob {a.get('outbreak_probability','N/A')}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Suggested Communication Messages", styles["Heading2"]))
    for msg in messages[:6]:
        story.append(
            Paragraph(
                f"[{msg.get('audience','').upper()}] {msg.get('city','N/A')}: {msg.get('message','')}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 4))

    doc.build(story)
    return buffer.getvalue()


def _map_disease_type(prognosis: str) -> str:
    p = prognosis.lower()
    for disease_type, keywords in DISEASE_TYPE_KEYWORDS.items():
        if any(k in p for k in keywords):
            return disease_type
    return "Other"


def disease_type_summary_from_weather(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not KAGGLE_PATH.exists():
        return []
    df = pd.read_csv(KAGGLE_PATH)
    if "prognosis" not in df.columns:
        return []

    disease_counts = (
        df["prognosis"]
        .astype(str)
        .value_counts()
        .rename_axis("disease")
        .reset_index(name="count")
    )
    disease_counts["type"] = disease_counts["disease"].apply(_map_disease_type)
    grouped = (
        disease_counts.groupby("type", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )

    top_per_type: dict[str, str] = {}
    for disease_type in grouped["type"]:
        top = disease_counts[disease_counts["type"] == disease_type].head(3)["disease"].tolist()
        top_per_type[disease_type] = ", ".join(top)

    # Weather-conditioned weighting from current internet-refreshed conditions.
    # Uses current district risk/weather signals rather than static global totals.
    avg_temp = 0.0
    avg_rain = 0.0
    avg_risk = 0.0
    if alerts:
        avg_temp = sum(float(a.get("temperature_c", 0.0)) for a in alerts) / len(alerts)
        avg_rain = sum(float(a.get("rainfall_mm", 0.0)) for a in alerts) / len(alerts)
        avg_risk = sum(float(a.get("risk_score", 0.0)) for a in alerts) / len(alerts)

    weights = {
        "Infectious": 1.0 + (0.5 if avg_rain >= 80 else 0.2) + (0.4 if avg_temp >= 28 else 0.1),
        "Respiratory": 1.0 + (0.35 if avg_temp <= 20 else 0.1) + (0.15 if avg_rain >= 60 else 0.05),
        "Gastrointestinal": 1.0 + (0.45 if avg_rain >= 90 else 0.2),
        "Cardiovascular": 1.0 + (0.45 if avg_temp >= 30 else 0.15),
        "Neurological": 1.0 + (0.2 if avg_temp >= 32 else 0.05),
        "Other": 1.0,
    }
    risk_multiplier = 1.0 + max(0.0, avg_risk - 0.35)

    result: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        disease_type = row["type"]
        weighted_count = int(round(row["count"] * weights.get(disease_type, 1.0) * risk_multiplier))
        result.append(
            {
                "type": disease_type,
                "count": weighted_count,
                "top_diseases": top_per_type.get(disease_type, ""),
            }
        )
    return sorted(result, key=lambda x: x["count"], reverse=True)


def disease_type_by_area(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return area-level disease-type risk view using current weather/risk.
    This is a practical prototype heuristic until district-level surveillance
    labels are integrated.
    """
    if not alerts:
        return []

    results: list[dict[str, Any]] = []
    for a in alerts:
        temp = float(a.get("temperature_c", 0.0))
        rain = float(a.get("rainfall_mm", 0.0))
        risk = float(a.get("risk_score", 0.0))

        scores = {
            "Infectious": 0.45 * min(1.0, rain / 120.0) + 0.35 * min(1.0, temp / 35.0) + 0.20 * risk,
            "Respiratory": 0.40 * max(0.0, (24.0 - temp) / 12.0) + 0.30 * min(1.0, rain / 100.0) + 0.30 * risk,
            "Gastrointestinal": 0.55 * min(1.0, rain / 140.0) + 0.20 * min(1.0, temp / 34.0) + 0.25 * risk,
            "Cardiovascular": 0.55 * min(1.0, temp / 36.0) + 0.15 * min(1.0, rain / 150.0) + 0.30 * risk,
            "Neurological": 0.50 * min(1.0, temp / 37.0) + 0.20 * min(1.0, rain / 160.0) + 0.30 * risk,
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[0]
        second = ranked[1]

        results.append(
            {
                "region": a.get("region", "N/A"),
                "city": a.get("city", a.get("district", "N/A")),
                "risk_level": a.get("risk_level", "low"),
                "dominant_type": top[0],
                "dominant_score": round(top[1], 3),
                "secondary_type": second[0],
                "secondary_score": round(second[1], 3),
            }
        )

    return sorted(results, key=lambda x: x["dominant_score"], reverse=True)
