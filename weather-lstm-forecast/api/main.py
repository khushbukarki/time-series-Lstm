"""Serve LSTM forecasts. Live endpoint pulls the latest observed days for Auckland from Open-Meteo."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import Config
from src.data import StandardScaler, add_features, fetch_recent
from src.model import LSTMForecaster

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
state: dict = {}


def load_artifacts(path: Path) -> dict:
    cfg = Config.from_dict(json.loads((path / "config.json").read_text()))
    model = LSTMForecaster(len(cfg.feature_names), cfg.horizon, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return {"cfg": cfg, "model": model, "scaler": StandardScaler.load(path / "scaler.json"),
            "metrics": json.loads((path / "metrics.json").read_text())}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if (ARTIFACTS / "model.pt").exists():
        state.update(load_artifacts(ARTIFACTS))
    yield
    state.clear()


app = FastAPI(title="LSTM Weather Forecast API", version="1.0.0", lifespan=lifespan)


def _require_model():
    if "model" not in state:
        raise HTTPException(503, "No trained model found. Run `python -m src.train` to create artifacts/.")


def _forecast(history: pd.DataFrame, days: int) -> list[dict]:
    cfg, scaler, model = state["cfg"], state["scaler"], state["model"]
    feats = add_features(history, cfg).tail(cfg.lookback)
    if len(feats) < cfg.lookback:
        raise HTTPException(422, f"Need {cfg.lookback} days of history, got {len(feats)}.")
    x = torch.from_numpy(scaler.transform(feats.to_numpy()).astype(np.float32))[None]
    with torch.no_grad():
        y = model(x).numpy()[0]
    t_idx = cfg.feature_names.index(cfg.target)
    temps = scaler.inverse_target(y, t_idx)[:days]
    start = feats.index[-1] + pd.Timedelta(days=1)
    return [{"date": str((start + pd.Timedelta(days=i)).date()), "mean_temp_c": round(float(t), 1)}
            for i, t in enumerate(temps)]


class HistoryRow(BaseModel):
    date: str
    temperature_2m_mean: float
    temperature_2m_max: float
    temperature_2m_min: float
    precipitation_sum: float = Field(ge=0)
    wind_speed_10m_max: float = Field(ge=0)


class PredictRequest(BaseModel):
    history: list[HistoryRow] = Field(min_length=1)
    days: int = Field(default=7, ge=1, le=7)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metrics")
def metrics():
    _require_model()
    return state["metrics"]


@app.get("/forecast")
def forecast(days: int = Query(7, ge=1, le=7)):
    """Live forecast for the configured location from the most recent observed days."""
    _require_model()
    cfg = state["cfg"]
    try:
        recent = fetch_recent(cfg, cfg.lookback + 5)
    except Exception as exc:  # network / upstream failure
        raise HTTPException(502, f"Could not fetch recent weather from Open-Meteo: {exc}") from exc
    return {"location": cfg.location, "based_on_data_to": str(recent.index[-1].date()),
            "forecast": _forecast(recent, days)}


@app.post("/predict")
def predict(req: PredictRequest):
    """Forecast from history you supply (at least `lookback` consecutive days)."""
    _require_model()
    df = pd.DataFrame([r.model_dump() for r in req.history])
    df["time"] = pd.to_datetime(df.pop("date"))
    return {"forecast": _forecast(df.set_index("time").sort_index(), req.days)}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LSTM forecast</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:2.5rem auto;padding:0 1rem;color:#1d2433}
table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{padding:.5rem;border-bottom:1px solid #dde2ea;text-align:left}
.m{color:#5b6475;font-size:.9rem}</style></head><body>
<h1>7-day temperature forecast from an LSTM</h1>
<p class="m">Daily mean temperature predicted from the last 30 days of observations. <a href="/docs">API docs</a></p>
<div id="out">Loading forecast…</div><div id="eval" class="m"></div>
<script>
fetch('/forecast').then(r=>r.json()).then(d=>{
 if(!d.forecast){document.getElementById('out').textContent=d.detail;return;}
 document.getElementById('out').innerHTML=`<p>${d.location}, using data to ${d.based_on_data_to}</p><table><tr><th>Date</th><th>Mean temperature</th></tr>`+
 d.forecast.map(f=>`<tr><td>${f.date}</td><td>${f.mean_temp_c} °C</td></tr>`).join('')+'</table>';});
fetch('/metrics').then(r=>r.json()).then(m=>{if(!m.day1)return;
 document.getElementById('eval').innerHTML='Next-day test RMSE: '+Object.entries(m.day1).map(([k,v])=>`${k} ${v.rmse.toFixed(2)} °C`).join(' · ');});
</script></body></html>"""
