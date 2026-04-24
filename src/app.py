from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request

from climate_data import fetch_climate_history
from service import (
    apply_simulation,
    build_trend_series,
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

app = Flask(__name__, template_folder=str(ROOT / "templates"))


@app.get("/")
def dashboard():
    df = pd.read_csv(SAMPLE_PATH)
    records = df.to_dict(orient="records")

    selected = request.args.get("risk_level", "").strip().lower()
    audience = request.args.get("audience", "").strip().lower()
    rainfall_factor = float(request.args.get("rainfall_factor", "1.0"))
    temp_offset = float(request.args.get("temp_offset", "0.0"))

    simulated = apply_simulation(records, rainfall_factor, temp_offset)
    alerts = score_records(simulated)
    if selected:
        alerts = [a for a in alerts if a["risk_level"] == selected]
    alerts = filter_by_audience(alerts, audience)

    counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    for alert in alerts:
        counts[alert["risk_level"]] += 1

    trends = build_trend_series(alerts)
    district_lookup = {a.get("district", ""): a for a in alerts}
    ai_summary = "AI/Data Science in use: weighted risk model, driver attribution, anomaly flags, trend estimation, and scenario simulation."

    return render_template(
        "dashboard.html",
        alerts=alerts,
        counts=counts,
        selected=selected,
        audience=audience,
        rainfall_factor=rainfall_factor,
        temp_offset=temp_offset,
        trends=trends,
        district_lookup=district_lookup,
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
    return redirect(request.referrer or "/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
