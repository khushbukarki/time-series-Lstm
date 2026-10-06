"""Download, feature engineering, chronological splits, scaling and windowing (NumPy/pandas only)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _to_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def download_history(cfg: Config, path: Path) -> pd.DataFrame:
    params = {
        "latitude": cfg.latitude, "longitude": cfg.longitude, "timezone": cfg.timezone,
        "start_date": cfg.start_date, "end_date": cfg.end_date, "daily": ",".join(cfg.weather_vars),
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = _to_frame(resp.json())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_recent(cfg: Config, days: int) -> pd.DataFrame:
    """Most recent `days` complete days, used by the API to make a live forecast."""
    params = {
        "latitude": cfg.latitude, "longitude": cfg.longitude, "timezone": cfg.timezone,
        "daily": ",".join(cfg.weather_vars), "past_days": days + 2, "forecast_days": 1,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = _to_frame(resp.json())
    today = pd.Timestamp.now(tz=cfg.timezone).normalize().tz_localize(None)
    return df[df.index < today].tail(days)


def load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fill small gaps and add a cyclical day-of-year encoding so the model knows the season."""
    out = df[list(cfg.weather_vars)].copy()
    out = out.asfreq("D").interpolate(limit=3).ffill().bfill()
    doy = out.index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def chronological_split(df: pd.DataFrame, cfg: Config):
    """Train / validation / test in time order - no shuffling, so no peeking at the future."""
    n = len(df)
    n_test, n_val = int(n * cfg.test_frac), int(n * cfg.val_frac)
    train = df.iloc[: n - n_val - n_test]
    # val/test keep `lookback` days of overlap so their first windows have full history
    val = df.iloc[n - n_val - n_test - cfg.lookback : n - n_test]
    test = df.iloc[n - n_test - cfg.lookback :]
    return train, val, test


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return cls(x.mean(axis=0), std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_target(self, y, target_idx: int = 0):
        return y * self.std[target_idx] + self.mean[target_idx]

    def save(self, path: Path):
        path.write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        d = json.loads(path.read_text())
        return cls(np.array(d["mean"]), np.array(d["std"]))


def make_windows(values: np.ndarray, lookback: int, horizon: int, target_idx: int = 0):
    """X: (n, lookback, n_features); y: (n, horizon) of the target column."""
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("series too short for the chosen lookback and horizon")
    X = np.stack([values[i : i + lookback] for i in range(n)])
    y = np.stack([values[i + lookback : i + lookback + horizon, target_idx] for i in range(n)])
    return X.astype(np.float32), y.astype(np.float32)


def synthetic_history(days: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Seasonal fake weather for tests and CI (no network needed)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=days, freq="D")
    season = 15 + 4.5 * np.cos(2 * np.pi * (idx.dayofyear - 30) / 365.25)
    noise = np.zeros(days)
    for t in range(1, days):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 1.2)
    mean = season + noise
    return pd.DataFrame({
        "temperature_2m_mean": mean,
        "temperature_2m_max": mean + 3 + rng.normal(0, 0.8, days),
        "temperature_2m_min": mean - 3 + rng.normal(0, 0.8, days),
        "precipitation_sum": np.clip(rng.gamma(0.6, 5, days) - 1, 0, None),
        "wind_speed_10m_max": rng.gamma(4, 5, days),
    }, index=pd.DatetimeIndex(idx, name="time"))
