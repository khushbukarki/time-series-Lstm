"""Shared settings for data, model and API."""
from dataclasses import asdict, dataclass


@dataclass
class Config:
    location: str = "Auckland"
    latitude: float = -36.85
    longitude: float = 174.76
    timezone: str = "Pacific/Auckland"
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-31"

    target: str = "temperature_2m_mean"
    weather_vars: tuple = (
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "wind_speed_10m_max",
    )
    lookback: int = 30          # days of history fed to the model
    horizon: int = 7            # days ahead predicted in one shot
    val_frac: float = 0.15
    test_frac: float = 0.15

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42

    def to_dict(self):
        d = asdict(self)
        d["weather_vars"] = list(self.weather_vars)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["weather_vars"] = tuple(d["weather_vars"])
        return cls(**d)

    @property
    def feature_names(self) -> list[str]:
        return list(self.weather_vars) + ["doy_sin", "doy_cos"]
