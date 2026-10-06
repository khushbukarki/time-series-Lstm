# Time-Series Weather Forecasting with LSTM Neural Networks

A PyTorch LSTM that forecasts Auckland's daily mean temperature 1–7 days ahead from the previous 30 days of weather, **benchmarked against persistence, moving-average and ARIMA baselines** on a held-out test period, and served through a **FastAPI** endpoint that makes live forecasts from the latest Open-Meteo observations.

**Live API:** `https://<your-service>.onrender.com` · **Docs:** `https://<your-service>.onrender.com/docs`

![Predictions on the test set](artifacts/test_predictions.png)

## Why baselines matter

Daily temperature is strongly autocorrelated, so "tomorrow = today" is already a decent forecast. A neural network is only worth deploying if it beats the simple methods on data it has never seen. This project measures exactly that.

## Method

| Step | Detail |
|---|---|
| Data | Daily mean/max/min temperature, precipitation and max wind for Auckland, 2014–2025, from the free [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api) |
| Features | The 5 weather variables plus sin/cos day-of-year so the model knows the season |
| Split | Chronological 70 / 15 / 15 train / validation / test. No shuffling across time; the scaler is fitted on training data only |
| Windows | 30-day input → next 7 days' mean temperature, predicted in one shot |
| Model | 2-layer LSTM (64 hidden units, dropout 0.2) → LayerNorm → linear head |
| Training | Adam, MSE loss, gradient clipping, LR reduced on plateau, early stopping on validation loss |
| Baselines | Persistence; 7-day moving average; ARIMA with (p,d,q) chosen by AIC on training data, then rolled forward with fixed parameters for genuine one-step-ahead forecasts |
| Metrics | RMSE and MAE in °C, overall and per forecast day |

## Results

After `python -m src.train`, results are printed and saved to `artifacts/metrics.json`. Fill in from your run:

| Model | Next-day RMSE (°C) | Next-day MAE (°C) |
|---|---|---|
| LSTM | | |
| ARIMA | | |
| Persistence | | |
| Moving average (7d) | | |

![Error by horizon](artifacts/rmse_by_horizon.png)

_Summarise the finding in a sentence or two, e.g. how much the LSTM improves on the best baseline at day 1, and how the gap changes further ahead._

## Run it

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is plenty here
pip install -r requirements.txt

python -m src.train --download     # downloads data, trains, evaluates, writes artifacts/ (a few minutes on CPU)
uvicorn api.main:app --reload      # http://localhost:8000
```

To forecast somewhere else, change `location`, `latitude` and `longitude` in `src/config.py` and retrain.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/forecast?days=7` | Live forecast from the latest observed days (Open-Meteo) |
| `POST` | `/predict` | Forecast from your own history: `{"history": [{date, temperature_2m_mean, ...} × 30], "days": 7}` |
| `GET` | `/metrics` | Test-set evaluation of the deployed model vs baselines |
| `GET` | `/health` | Service and model status |

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers window alignment (targets start the day after each window), chronological splitting without leakage, scaler round-trips, baseline maths, an ARIMA run, and an end-to-end train-then-serve smoke test on synthetic data.

## Deploy (free on Render)

1. Train locally so `artifacts/` holds `model.pt`, `scaler.json`, `config.json` and `metrics.json`, then commit and push them (they are small).
2. On [render.com](https://render.com) create a **New Web Service** from this repo. Render detects the `Dockerfile`; choose the free instance type.
3. Once it's live, put the URL at the top of this README. The free tier sleeps when idle, so the first request after a while takes ~30 seconds.

## Project structure

```
src/config.py      all settings in one place
src/data.py        download, features, chronological split, scaling, windowing
src/model.py       LSTM model
src/baselines.py   persistence, moving average, ARIMA
src/train.py       training, evaluation, plots, artifacts
api/main.py        FastAPI service
tests/             pytest suite
```
