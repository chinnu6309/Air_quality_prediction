# Data for the final project

Use actual observed data for the final report, not `generate_demo_data.py` output.

Recommended scope: use the **Air Quality Data in India (2015–2020)** dataset and train a model for one city, for example Delhi. Download `city_day.csv` from its [Kaggle dataset page](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india/data) and save it in this folder.

The data contains daily readings from air-quality monitoring stations in Indian cities. Its fields include `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, and `AQI`, which match the training program. The official data origin is the Central Pollution Control Board (CPCB) monitoring network. The CPCB dashboard reports that the network provides station- and city-level AQI, and its station list identifies locations by state, city, station name, agency, and site code.

For a city-specific model, run:

```bash
python train.py --data data/city_day.csv --city Delhi
```

In your report, state the exact city selected, the date range of the rows remaining after filtering, source link, and download date. Do not say the readings were collected personally: they were collected by monitoring stations and used by your project.
