import numpy as np
import pytest

from src.baselines import moving_average, persistence
from src.config import Config
from src.data import StandardScaler, add_features, chronological_split, make_windows, synthetic_history
from src.metrics import mae, rmse


def test_windows_shape_and_alignment():
    values = np.arange(20, dtype=float).reshape(10, 2)  # col 0 = 0,2,4,...
    X, y = make_windows(values, lookback=3, horizon=2, target_idx=0)
    assert X.shape == (6, 3, 2) and y.shape == (6, 2)
    assert X[0, -1, 0] == 4 and y[0].tolist() == [6, 8]  # targets start right after the window


def test_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.zeros((5, 1)), lookback=4, horizon=2)


def test_split_is_chronological_without_leakage():
    cfg = Config()
    feats = add_features(synthetic_history(1000), cfg)
    train, val, test = chronological_split(feats, cfg)
    # the only overlap allowed is the lookback context; first val/test *targets* come after train ends
    assert val.index[cfg.lookback] > train.index[-1]
    assert test.index[cfg.lookback] > val.index[-1] or test.index[cfg.lookback] == val.index[-1] + np.timedelta64(1, "D")


def test_scaler_roundtrip(tmp_path):
    x = np.random.default_rng(0).normal(5, 2, size=(100, 3))
    s = StandardScaler.fit(x)
    assert np.allclose(s.transform(x).mean(axis=0), 0, atol=1e-9)
    s.save(tmp_path / "s.json")
    s2 = StandardScaler.load(tmp_path / "s.json")
    assert np.allclose(s2.inverse_target(s2.transform(x)[:, 0]), x[:, 0])


def test_features_include_season():
    feats = add_features(synthetic_history(400), Config())
    assert {"doy_sin", "doy_cos"} <= set(feats.columns)
    assert not feats.isna().any().any()


def test_baselines_and_metrics():
    hist = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert persistence(hist, 3).tolist() == [[7.0, 7.0, 7.0]]
    assert moving_average(hist, 2, window=7).tolist() == [[4.0, 4.0]]
    assert rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
    assert mae([1, 2], [1, 4]) == 1.0
