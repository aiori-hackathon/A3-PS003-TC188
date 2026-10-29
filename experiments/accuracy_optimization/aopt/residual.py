"""Residual correction for a base forecast: LightGBM on features known before the forecast starts.

Features per (window, horizon) cell, all from steps < t0 or the calendar of the target step:
  base forecast, horizon, last raw reading (t0-1), mean of the last 12 raw readings, 3-step trend (t0-1 minus t0-4),
  5-minute slot of the target, weekday of the target, sensor position (categorical).
Missing past readings stay NaN (LightGBM routes them); no target or later value is ever read.
"""

import warnings

import numpy as np

FEATURES = ["base", "horizon", "last", "mean12", "trend3", "slot", "weekday", "sensor"]
PARAMS = dict(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def features(pred, idx, flow):
    """(N, 12, 8) features for base forecasts `pred` (N, 12) on windows idx (N, 2) [sensor, t0]; flow (S, T) raw."""
    s, t0 = idx[:, 0], idx[:, 1]
    last = flow[s, t0 - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean12 = np.nanmean(flow[s[:, None], t0[:, None] + np.arange(-12, 0)], 1)
    trend = flow[s, t0 - 1] - flow[s, t0 - 4]
    n = len(idx)
    out = np.empty((n, 12, len(FEATURES)))
    for h in range(12):
        t = t0 + h
        out[:, h] = np.column_stack([pred[:, h], np.full(n, h), last, mean12, trend, t % 288, (6 + t // 288) % 7, s])
    return out


def fit(X, resid, mask, seed=0, max_rows=3_000_000):
    import lightgbm as lgb
    Xf, rf = X.reshape(-1, X.shape[-1]), resid.ravel()
    keep = np.nonzero(mask.ravel())[0]
    keep = np.random.default_rng(seed).choice(keep, min(max_rows, len(keep)), replace=False)
    model = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    model.fit(Xf[keep], rf[keep], categorical_feature=[FEATURES.index("sensor")])
    return model


def apply(model, pred, X):
    """Corrected forecast, clipped at 0 (the frozen policy)."""
    return np.maximum(pred + model.predict(X.reshape(-1, X.shape[-1])).reshape(pred.shape), 0)
