"""Predict AQI for one record or a CSV of records using the saved model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train import ROOT, make_features


def predict_frame(raw: pd.DataFrame) -> pd.DataFrame:
    model_path = ROOT / "artifacts" / "air_quality_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Saved model not found. Run `python train.py` first.")
    bundle = joblib.load(model_path)
    X, _, _ = make_features(raw, require_target=False)
    result = raw.copy()
    result["Predicted_AQI"] = bundle["model"].predict(X).round(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AQI with the saved air-quality model.")
    parser.add_argument("--input", help="CSV file to predict. It must have Date and the nine pollutant/weather columns.")
    parser.add_argument("--output", default="artifacts/new_predictions.csv", help="Where to save CSV predictions.")
    parser.add_argument("--date", help="Single-reading date, e.g. '2025-06-15 14:00:00'.")
    for feature in ("pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"):
        parser.add_argument(f"--{feature}", type=float)
    args = parser.parse_args()

    if args.input:
        raw = pd.read_csv(args.input)
    else:
        values = vars(args)
        needed = ["date", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"]
        missing = [name for name in needed if values[name] is None]
        if missing:
            parser.error("Provide --input, or all single-reading values. Missing: " + ", ".join(missing))
        raw = pd.DataFrame([{
            "Date": args.date, "PM2.5": args.pm25, "PM10": args.pm10, "NO2": args.no2,
            "SO2": args.so2, "CO": args.co, "O3": args.o3, "Temperature": args.temperature,
            "Humidity": args.humidity, "WindSpeed": args.wind_speed,
        }])

    result = predict_frame(raw)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result[["Predicted_AQI"]].to_string(index=False))
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
