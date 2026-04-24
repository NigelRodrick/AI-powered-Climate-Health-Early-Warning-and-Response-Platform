# AI-powered Climate-Health Early Warning and Response Platform

An open-source MVP that demonstrates how climate and health data can be combined to produce localized child-health risk alerts and recommended response actions.

## Why this project

Climate shocks such as flooding and heat waves can rapidly increase health risks for children. Frontline actors often receive late or fragmented signals. This prototype shows a lightweight approach to:

- ingest climate and health indicators,
- compute a transparent risk score,
- classify risk levels, and
- generate actionable alerts for communities, schools, and clinics.

## MVP scope

This repository includes:

- data ingestion and feature preparation (`src/data_ingestion.py`)
- rule-based baseline risk scoring (`src/risk_scoring.py`)
- alert payload generation (`src/alerting.py`)
- a runnable end-to-end demo (`src/main.py`)
- a Flask API + dashboard (`src/app.py`)
- sample input data (`sample_data/climate_health_sample.csv`)
- basic tests (`tests/test_risk_scoring.py`)

## Quick start

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the prototype:

```bash
python src/main.py
```

4. Review outputs in `output/alerts.json`.

## Run web demo

Start the API and dashboard:

```bash
python src/app.py
```

Then open:

- `http://localhost:8000/` dashboard
- `http://localhost:8000/health` health check

## Train ML outbreak model with internet climate data

This project can fetch historical climate conditions from Open-Meteo and train a district-level outbreak classifier.

```bash
python src/train_model.py
```

Generated artifacts:

- `models/outbreak_model.joblib` trained model
- `models/model_report.txt` training summary
- `sample_data/training_dataset.csv` assembled training data

From the dashboard:

- **Refresh Climate From Internet** updates recent temperature/rainfall in sample records.
- **Train Outbreak Model** trains and activates model-based inference for live scoring.

### Example API request

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d "{\"records\":[{\"location_id\":\"NG-ABJ-100\",\"week\":\"2026-W17\",\"rainfall_mm\":140,\"temperature_c\":35,\"flood_risk_index\":0.7,\"clinic_case_load_index\":0.6,\"medicine_stock_gap_index\":0.4}]}"
```

## Data model

Each record represents a location and week-level indicators:

- `location_id`
- `week`
- `rainfall_mm`
- `temperature_c`
- `flood_risk_index` (0 to 1)
- `clinic_case_load_index` (0 to 1)
- `medicine_stock_gap_index` (0 to 1)

## Open-source approach

This MVP is released under the MIT License. See `docs/open-source-strategy.md` for intended licensing and governance practices for scale-up phases.

## Disclaimer

This is a prototype for decision support and not a replacement for clinical judgment or emergency command protocols.
