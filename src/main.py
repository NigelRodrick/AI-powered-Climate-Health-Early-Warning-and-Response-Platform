from __future__ import annotations

import json
from pathlib import Path

from data_ingestion import load_input_data, normalize_features
from risk_scoring import score_locations
from alerting import build_alerts


def run() -> Path:
    root = Path(__file__).resolve().parent.parent
    input_path = root / "sample_data" / "climate_health_sample.csv"
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "alerts.json"

    df = load_input_data(input_path)
    df = normalize_features(df)
    scored = score_locations(df)
    alerts = build_alerts(scored)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    print(f"Generated {len(alerts)} alerts at {output_path}")
    return output_path


if __name__ == "__main__":
    run()
