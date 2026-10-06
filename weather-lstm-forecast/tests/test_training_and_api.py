"""Smoke test: train briefly on synthetic data, then serve it through the API."""
import json

import pytest

torch = pytest.importorskip("torch")

from src.baselines import arima_one_step  # noqa: E402
from src.config import Config  # noqa: E402
from src.data import synthetic_history  # noqa: E402
from src.train import run  # noqa: E402


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = Config(max_epochs=3, patience=2)
    metrics = run(cfg, synthetic_history(900), out, max_epochs=3, skip_arima=True, log=lambda *_: None)
    return out, metrics


def test_training_writes_artifacts(artifacts):
    out, metrics = artifacts
    for f in ["model.pt", "scaler.json", "config.json", "metrics.json", "test_predictions.png"]:
        assert (out / f).exists(), f
    assert set(metrics["day1"]) >= {"LSTM", "Persistence", "Moving average (7d)"}
    assert len(metrics["per_horizon"]["LSTM"]["rmse"]) == Config().horizon


def test_arima_baseline_runs():
    pytest.importorskip("statsmodels")
    series = synthetic_history(400)["temperature_2m_mean"].to_numpy()
    preds, order = arima_one_step(series, 300, 330, 50, orders=[(1, 0, 0), (2, 0, 1)])
    assert len(preds) == 50 and order in [(1, 0, 0), (2, 0, 1)]


def test_api_predict(artifacts, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api

    out, _ = artifacts
    monkeypatch.setattr(api, "ARTIFACTS", out)
    hist = synthetic_history(40).reset_index()
    rows = [{"date": str(r.time.date()), **{k: float(getattr(r, k)) for k in Config().weather_vars}}
            for r in hist.itertuples()]
    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        body = client.post("/predict", json={"history": rows, "days": 3}).json()
        assert len(body["forecast"]) == 3
        assert all(-10 < f["mean_temp_c"] < 40 for f in body["forecast"])
        assert client.post("/predict", json={"history": rows[:5]}).status_code == 422
        assert "day1" in client.get("/metrics").json()
    json.dumps(body)
