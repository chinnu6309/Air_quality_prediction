"""Train, evaluate, and save an air-quality AQI regression model.

Expected CSV columns:
Date, PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed, AQI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
TARGET = "AQI"
RAW_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]
TIME_FEATURES = ["Year", "Month", "Day", "DayOfWeek", "Hour"]
MODEL_FEATURES = RAW_FEATURES + TIME_FEATURES


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept common spelling/case variations while saving standard column names."""
    aliases = {
        "date": "Date", "datetime": "Date", "timestamp": "Date",
        "pm25": "PM2.5", "pm2.5": "PM2.5", "pm_2_5": "PM2.5",
        "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3",
        "temperature": "Temperature", "temp": "Temperature",
        "humidity": "Humidity", "windspeed": "WindSpeed", "wind_speed": "WindSpeed",
        "aqi": "AQI", "airqualityindex": "AQI", "air_quality_index": "AQI",
    }
    rename_map = {}
    for name in frame.columns:
        key = str(name).strip().lower().replace(" ", "_")
        compact_key = key.replace("_", "")
        if key in aliases:
            rename_map[name] = aliases[key]
        elif compact_key in aliases:
            rename_map[name] = aliases[compact_key]
    return frame.rename(columns=rename_map)


def make_features(frame: pd.DataFrame, require_target: bool) -> Tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Clean raw data and derive numeric date features without leaking the target."""
    frame = normalize_columns(frame.copy())
    required = ["Date", *RAW_FEATURES]
    if require_target:
        required.append(TARGET)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{dates.isna().sum()} invalid date value(s) found in the Date column.")

    features = frame[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    features["Year"] = dates.dt.year
    features["Month"] = dates.dt.month
    features["Day"] = dates.dt.day
    features["DayOfWeek"] = dates.dt.dayofweek
    features["Hour"] = dates.dt.hour

    target = None
    if require_target:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        valid_target = target.notna()
        dropped = int((~valid_target).sum())
        if dropped:
            print(f"Dropped {dropped} row(s) with missing/non-numeric AQI target values.")
        features, target, dates = features.loc[valid_target], target.loc[valid_target], dates.loc[valid_target]
    return features.reset_index(drop=True), None if target is None else target.reset_index(drop=True), dates.reset_index(drop=True)


def build_pipeline(model: object) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, MODEL_FEATURES)], remainder="drop"
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R2": float(r2_score(y_true, prediction)),
    }


def save_plot(y_test: pd.Series, prediction: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_test, prediction, alpha=0.7, color="#147D9A", edgecolors="none")
    lower = min(y_test.min(), prediction.min())
    upper = max(y_test.max(), prediction.max())
    ax.plot([lower, upper], [lower, upper], "--", color="#E05D44", label="Perfect prediction")
    ax.set(xlabel="Actual AQI", ylabel="Predicted AQI", title="Air Quality Model: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def train(data_path: Path, test_size: float, city: str | None = None) -> dict:
    if not 0.05 <= test_size < 0.5:
        raise ValueError("test_size must be at least 0.05 and less than 0.5")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    if city:
        city_column = next((column for column in raw.columns if str(column).strip().lower() == "city"), None)
        if city_column is None:
            raise ValueError("--city was supplied but the CSV has no City column.")
        raw = raw[raw[city_column].astype(str).str.strip().str.casefold() == city.strip().casefold()].copy()
        if raw.empty:
            raise ValueError(f"No records found for city: {city}")
        print(f"Using {len(raw)} records collected in {city}.")
    X, y, dates = make_features(raw, require_target=True)
    assert y is not None
    if len(X) < 30:
        raise ValueError("Use at least 30 valid rows so training and testing are meaningful.")

    # Chronological split mimics predicting future measurements from past measurements.
    order = dates.sort_values().index
    X, y = X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.04, max_depth=3),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=300, min_samples_leaf=2, n_jobs=-1),
    }
    fitted_models: dict[str, Pipeline] = {}
    results: list[dict[str, float | str]] = []
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        result = {"model": name, **metrics(y_test, pipeline.predict(X_test))}
        fitted_models[name] = pipeline
        results.append(result)

    results.sort(key=lambda item: float(item["R2"]), reverse=True)
    best_name = str(results[0]["model"])
    best_model = fitted_models[best_name]
    prediction = best_model.predict(X_test)
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    pd.DataFrame({"Actual_AQI": y_test, "Predicted_AQI": prediction, "Residual": y_test - prediction}).to_csv(
        ARTIFACTS / "test_predictions.csv", index=False
    )
    save_plot(y_test, prediction)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": MODEL_FEATURES,
        "raw_feature_names": RAW_FEATURES,
        "target_name": TARGET,
        "metrics": results[0],
    }
    joblib.dump(bundle, ARTIFACTS / "air_quality_model.joblib")
    report = {
        "dataset": str(data_path), "city_filter": city, "rows_after_target_cleaning": int(len(X)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "missing_feature_values_before_imputation": {key: int(value) for key, value in X.isna().sum().items()},
        "best_model": best_name, "best_test_metrics": results[0], "all_models": results,
    }
    (ARTIFACTS / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AQI regression model.")
    parser.add_argument("--data", default="data/air_quality.csv", help="Path to the CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Chronological test proportion (default: 0.20).")
    parser.add_argument("--city", help="Optional City value to train a location-specific model.")
    args = parser.parse_args()
    report = train(Path(args.data), args.test_size, args.city)
    print(f"Best model: {report['best_model']}")
    print("Test metrics:", report["best_test_metrics"])
    print("Saved model: artifacts/air_quality_model.joblib")


if __name__ == "__main__":
    main()
