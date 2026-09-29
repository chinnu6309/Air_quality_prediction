# Air Quality Prediction — Regression Project

This project predicts a **continuous AQI value** from pollutant, weather, and time readings. It includes data preprocessing, a chronological train/test split, comparison of three regression models, saved artifacts, command-line prediction, and a Flask prediction form.

## Dataset: what to say in the presentation

Use the `city_day.csv` file described in [`data/README.md`](data/README.md), then run the model for one city (for example, Delhi).

> “The observations were collected by continuous air-quality monitoring stations in **Delhi, India** (after filtering the India-wide dataset to `City = Delhi`). The station network is managed by CPCB and partner state agencies. Each row contains a date, pollutant concentrations and the reported AQI. I used readings from [state the first and last date in my CSV].”

This is accurate only after you actually train with `--city Delhi`; substitute your own city if different. The official [CPCB National AQI dashboard](https://app.cpcbccr.com/AQIIndia) explains that station data supports city- and station-level AQI. Its [station reference](https://app.cpcbccr.com/ccr_docs/caaqms_list_All_India.pdf) lists station locations and operating agencies.

## Why this is regression

The target `AQI` is a number, not a category. The model learns to estimate its numeric value; for example, 173.4. The project evaluates MAE, RMSE, and R², rather than classification accuracy.

## Best-model rule (important)

There is no honest universal “best model” before testing. `train.py` trains **Ridge Regression**, **Gradient Boosting**, and **Random Forest** on the same training rows, then selects the one with the **highest test R²**. It also reports RMSE and MAE. The winner is saved as `artifacts/air_quality_model.joblib`.

Usually, tree-based models can be strong here because the AQI relationship with pollutants is nonlinear and pollutants interact. Ridge is included as an interpretable linear baseline. But your conclusion must use your generated `artifacts/model_comparison.csv`: say `MODEL_NAME` is best because it has the highest unseen-test R² and the lowest (or suitably low) RMSE/MAE among the candidates. Do not select a model only because it had the best training score.

## Setup and run

```bash
cd "/Users/chegireddypandurangareddy/Documents/New project/air_quality_regression"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an end-to-end demonstration only:

```bash
python generate_demo_data.py
python train.py
```

For the final project, download `city_day.csv` as explained in `data/README.md` and train one location:

```bash
python train.py --data data/city_day.csv --city Delhi
```

The workflow does the following:

1. Converts dates into year/month/day/day-of-week/hour features.
2. Converts inputs to numeric values and replaces missing feature values with the **training-set median**.
3. Sorts data by date; trains on the oldest 80% and tests on the newest 20%, so future rows are not used to predict the past.
4. Compares models, saves test predictions, the model, a JSON report, and an actual-vs-predicted chart in `artifacts/`.

## Make predictions

Single reading:

```bash
python predict.py --date "2025-06-15 14:00:00" --pm25 86 --pm10 145 --no2 42 --so2 9 --co 1.1 --o3 34 --temperature 31 --humidity 60 --wind_speed 2.4
```

For many readings, prepare a CSV with all input columns except `AQI` and run:

```bash
python predict.py --input data/new_readings.csv --output artifacts/new_predictions.csv
```

Or start the web form after training:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Report conclusion template

> I predicted the numerical Air Quality Index (AQI) using daily pollutant and weather readings collected by air-quality monitoring stations in [CITY], India. After date-feature engineering, numeric conversion, median imputation, and a chronological 80:20 train/test split, I compared Ridge Regression, Gradient Boosting, and Random Forest. [WINNER] was selected because it gave the highest R² of [VALUE] on the unseen test set, with MAE [VALUE] and RMSE [VALUE]. This means its estimates were, on average, about [MAE] AQI units away from the observed AQI. The result applies to the selected place and date range; it should not be treated as a public-health forecast without ongoing validation.
