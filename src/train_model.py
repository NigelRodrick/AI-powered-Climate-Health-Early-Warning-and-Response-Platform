from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from climate_data import build_weekly_features, default_date_range, fetch_climate_history


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "outbreak_model.joblib"
TRAINING_SET_PATH = ROOT / "sample_data" / "training_dataset.csv"
KAGGLE_PATH = ROOT / "sample_data" / "kaggle_disease_dataset.csv"
REPORT_PATH = ROOT / "models" / "model_report.txt"


def add_proxy_outbreak_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy target while real surveillance labels are being integrated.
    Use climate stress thresholds as initial weak supervision.
    """
    out = df.copy()
    out["outbreak_next_2w"] = (
        (out["temperature_c"] > 33.0) | (out["rainfall_mm"] > out["rainfall_mm"].quantile(0.7))
    ).astype(int)
    return out


def train_from_kaggle(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, str, list[str]]:
    """
    Train on Kaggle disease records.
    Positive class approximates infectious/outbreak-relevant conditions.
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    if "prognosis" not in out.columns:
        raise ValueError("Kaggle dataset missing required 'prognosis' column.")

    text = out["prognosis"].astype(str).str.lower()
    infectious_keywords = [
        "flu",
        "influenza",
        "cold",
        "malaria",
        "dengue",
        "cholera",
        "covid",
        "pneumonia",
        "typhoid",
        "tuberculosis",
        "hepatitis",
        "measles",
        "diarrhea",
        "infection",
    ]
    out["outbreak_next_2w"] = text.apply(lambda v: int(any(k in v for k in infectious_keywords)))

    feature_cols = [c for c in out.columns if c != "prognosis" and c != "outbreak_next_2w"]
    X = out[feature_cols]
    y = out["outbreak_next_2w"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_cols,
            )
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=600)),
        ]
    )
    model.fit(X, y)
    preds = model.predict(X)
    report = classification_report(y, preds, digits=3)
    return model, out, report, feature_cols


def train(start_date: str | None = None, end_date: str | None = None) -> Path:
    training_source = "internet_climate_proxy"
    if KAGGLE_PATH.exists():
        kaggle = pd.read_csv(KAGGLE_PATH)
        model, weekly, report, feature_cols = train_from_kaggle(kaggle)
        training_source = "kaggle_disease_dataset"
    else:
        if not start_date or not end_date:
            start_date, end_date = default_date_range()
        climate_df = fetch_climate_history(start_date, end_date)
        weekly = build_weekly_features(climate_df)
        weekly = add_proxy_outbreak_label(weekly)

        feature_cols = ["temperature_c", "rainfall_mm", "temp_anomaly", "rain_anomaly", "flood_risk_index"]
        X = weekly[feature_cols]
        y = weekly["outbreak_next_2w"]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    feature_cols,
                )
            ]
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        model.fit(X, y)
        preds = model.predict(X)
        report = classification_report(y, preds, digits=3)

    MODEL_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model,
        "training_source": training_source,
        "feature_columns": feature_cols,
    }
    joblib.dump(artifact, MODEL_PATH)
    weekly.to_csv(TRAINING_SET_PATH, index=False)
    REPORT_PATH.write_text(
        f"Training source: {training_source}\nRows: {len(weekly)}\n\n{report}",
        encoding="utf-8",
    )
    return MODEL_PATH


if __name__ == "__main__":
    path = train()
    print(f"Saved model to: {path}")
