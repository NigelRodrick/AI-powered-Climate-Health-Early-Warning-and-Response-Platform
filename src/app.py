from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request

from climate_data import fetch_climate_history
from communication import generate_message_templates
from service import (
    LOCATION_COORDS,
    apply_simulation,
    build_trend_series,
    disease_type_by_area,
    disease_type_summary_from_weather,
    export_daily_brief,
    export_csv,
    export_pdf,
    filter_by_audience,
    score_records,
    score_sample_file,
    update_action_status,
)
from train_model import train


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "sample_data" / "climate_health_sample.csv"
STATE_PATH = ROOT / "sample_data" / "live_state.json"

app = Flask(__name__, template_folder=str(ROOT / "templates"))


@app.get("/")
def dashboard():
    state = read_live_state()
    demo_mode = bool(state.get("demo_mode", False))
    if not demo_mode:
        ensure_recent_climate()
    df = pd.read_csv(SAMPLE_PATH)
    records = df.to_dict(orient="records")

    selected = request.args.get("risk_level", "").strip().lower()
    audience = request.args.get("audience", "").strip().lower()
    selected_region = request.args.get("region", "").strip()
    selected_city = request.args.get("city", "").strip()
    rainfall_factor = float(request.args.get("rainfall_factor", "1.0"))
    temp_offset = float(request.args.get("temp_offset", "0.0"))

    simulated = apply_simulation(records, rainfall_factor, temp_offset)
    all_alerts = score_records(simulated)
    if selected_region:
        all_alerts = [a for a in all_alerts if a.get("region", "") == selected_region]
    if selected_city:
        all_alerts = [a for a in all_alerts if a.get("city", "") == selected_city]

    alerts = list(all_alerts)
    if selected:
        alerts = [a for a in alerts if a["risk_level"] == selected]
    alerts = filter_by_audience(alerts, audience)

    counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    for alert in all_alerts:
        counts[alert["risk_level"]] += 1

    trends = build_trend_series(all_alerts)
    district_lookup = {a.get("district", ""): a for a in all_alerts}
    disease_types = disease_type_summary_from_weather(all_alerts)
    disease_types_area = disease_type_by_area(all_alerts)
    risk_alerts = [a for a in all_alerts if a["risk_score"] >= 0.55]
    message_templates = generate_message_templates(risk_alerts)
    region_options = sorted({v.get("region", "") for v in LOCATION_COORDS.values()})
    city_options = sorted(
        {
            v.get("city", "")
            for v in LOCATION_COORDS.values()
            if (not selected_region or v.get("region", "") == selected_region)
        }
    )
    ai_summary = "AI/Data Science in use: weighted risk model, driver attribution, anomaly flags, trend estimation, and scenario simulation."
    last_updated = state.get("last_updated_utc", "N/A")
    top_priority_alerts = sorted(risk_alerts, key=lambda a: float(a["risk_score"]), reverse=True)[:3]

    return render_template(
        "dashboard.html",
        all_alerts=all_alerts,
        alerts=alerts,
        counts=counts,
        selected=selected,
        audience=audience,
        selected_region=selected_region,
        selected_city=selected_city,
        region_options=region_options,
        city_options=city_options,
        rainfall_factor=rainfall_factor,
        temp_offset=temp_offset,
        trends=trends,
        district_lookup=district_lookup,
        disease_types=disease_types,
        disease_types_area=disease_types_area,
        risk_alerts=risk_alerts,
        top_priority_alerts=top_priority_alerts,
        demo_mode=demo_mode,
        message_templates=message_templates,
        last_updated=last_updated,
        ai_summary=ai_summary,
    )


@app.post("/score")
def score():
    payload = request.get_json(silent=True) or {}
    records = payload.get("records", [])
    alerts = score_records(records)
    return jsonify({"alerts": alerts, "count": len(alerts)})


@app.post("/action-status")
def action_status():
    location_id = request.form.get("location_id", "")
    status = request.form.get("status", "new")
    if location_id:
        update_action_status(location_id, status)
    return redirect(request.referrer or "/")


@app.get("/export.csv")
def export_alerts_csv():
    alerts = score_sample_file(SAMPLE_PATH)
    csv_content = export_csv(alerts)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=dashboard_alerts.csv"},
    )


@app.get("/export.pdf")
def export_alerts_pdf():
    alerts = score_sample_file(SAMPLE_PATH)
    pdf_bytes = export_pdf(alerts)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=dashboard_alerts.pdf"},
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/train-model")
def train_model():
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    model_path = train(start_date=start_date, end_date=end_date)
    return jsonify({"status": "trained", "model_path": str(model_path)})


@app.post("/refresh-climate")
def refresh_climate():
    last_updated = refresh_climate_dataset()
    state = read_live_state()
    state["last_updated_utc"] = last_updated
    write_live_state(state)
    return redirect(request.referrer or "/")


@app.post("/refresh-climate-json")
def refresh_climate_json():
    last_updated = refresh_climate_dataset()
    state = read_live_state()
    state["last_updated_utc"] = last_updated
    write_live_state(state)
    return jsonify({"status": "ok", "last_updated_utc": last_updated})


@app.get("/live-predictions")
def live_predictions():
    df = pd.read_csv(SAMPLE_PATH)
    alerts = score_records(df.to_dict(orient="records"))
    return jsonify(
        {
            "status": "ok",
            "last_updated_utc": read_live_state().get("last_updated_utc", "N/A"),
            "predictions": alerts,
        }
    )


@app.post("/run-demo-cycle")
def run_demo_cycle():
    last_updated = refresh_climate_dataset()
    state = read_live_state()
    state["last_updated_utc"] = last_updated
    write_live_state(state)
    return redirect(request.referrer or "/")


@app.post("/set-demo-mode")
def set_demo_mode():
    enabled = request.form.get("enabled", "0") == "1"
    state = read_live_state()
    state["demo_mode"] = enabled
    write_live_state(state)
    return redirect(request.referrer or "/")


@app.get("/export-daily-brief.pdf")
def export_daily_brief_pdf():
    df = pd.read_csv(SAMPLE_PATH)
    alerts = score_records(df.to_dict(orient="records"))
    messages = generate_message_templates([a for a in alerts if a["risk_score"] >= 0.55])
    pdf_bytes = export_daily_brief(alerts, messages)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=daily_risk_brief.pdf"},
    )


def refresh_climate_dataset() -> str:
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    climate = fetch_climate_history(start_date=start_date, end_date=end_date)
    latest = (
        climate.sort_values("date")
        .groupby(["location_id", "district"], as_index=False)
        .tail(1)[["location_id", "temperature_c", "rainfall_mm"]]
    )
    base = pd.read_csv(SAMPLE_PATH)
    merged = base.drop(columns=["temperature_c", "rainfall_mm"]).merge(latest, on="location_id", how="left")
    merged["temperature_c"] = merged["temperature_c"].fillna(base["temperature_c"])
    merged["rainfall_mm"] = merged["rainfall_mm"].fillna(base["rainfall_mm"])
    merged.to_csv(SAMPLE_PATH, index=False)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_recent_climate(max_age_minutes: int = 30) -> None:
    state = read_live_state()
    ts = state.get("last_updated_utc")
    if not ts:
        last_updated = refresh_climate_dataset()
        write_live_state({"last_updated_utc": last_updated})
        return
    try:
        seen = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except ValueError:
        last_updated = refresh_climate_dataset()
        write_live_state({"last_updated_utc": last_updated})
        return
    age_min = (datetime.now(timezone.utc) - seen).total_seconds() / 60.0
    if age_min > max_age_minutes:
        last_updated = refresh_climate_dataset()
        write_live_state({"last_updated_utc": last_updated})


def read_live_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_live_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
