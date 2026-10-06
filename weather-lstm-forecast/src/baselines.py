"""Baselines the LSTM has to beat. All predict the same targets as the LSTM test windows."""
import itertools
import warnings

import numpy as np


def persistence(X_target: np.ndarray, horizon: int) -> np.ndarray:
    """Tomorrow (and every later day) = today. X_target: (n, lookback) unscaled target history."""
    return np.repeat(X_target[:, -1:], horizon, axis=1)


def moving_average(X_target: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    return np.repeat(X_target[:, -window:].mean(axis=1, keepdims=True), horizon, axis=1)


def arima_one_step(series: np.ndarray, train_end: int, first_target: int, n_windows: int,
                   orders=None) -> tuple[np.ndarray, tuple]:
    """Fit ARIMA on series[:train_end] (order chosen by AIC), then roll forward through the rest of the
    series with the fitted parameters fixed, giving genuine one-step-ahead forecasts (no refit, no
    look-ahead). Returns forecasts for series[first_target : first_target + n_windows]."""
    from statsmodels.tsa.arima.model import ARIMA

    train, rest = series[:train_end], series[train_end:]
    orders = orders or list(itertools.product([1, 2, 3], [0, 1], [0, 1, 2]))
    best, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                res = ARIMA(train, order=order).fit()
            except Exception:
                continue
            if res.aic < best_aic:
                best, best_aic = (order, res), res.aic
        order, res = best
        extended = res.append(rest, refit=False)
    # fittedvalues[t] is the forecast of series[t] made with data up to t-1
    one_step = np.asarray(extended.fittedvalues)
    return one_step[first_target : first_target + n_windows], order
