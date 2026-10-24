"""Leak-safe preprocessing for (sensor, time) flow series.

Rules enforced here:
- NaN means missing; zero is a valid observation.
- Every fitted statistic (scaler, time-of-day median, fallback mean) uses only
  observed values inside the training split.
- Input imputation is causal: forward fill (bounded, never across a split boundary),
  then the training time-of-day median, then the training mean. Linear interpolation
  was rejected because its right endpoint can be a future/target value.
- Targets are never imputed; they keep NaN and are masked downstream.
"""

import warnings

import numpy as np

from aiori.config import STEPS_PER_DAY

MIN_STD = 1e-6


def observed_mask(x):
    return np.isfinite(x)


def _train(x, splits):
    start, end = splits["train"]
    return np.asarray(x[:, start:end], dtype=np.float64)


def fit_scaler(x, splits, mode="per_sensor"):
    """Z-score parameters from training-split observations only.

    per_sensor: each sensor's own mean/std (computable locally by the client holding it).
    global: one mean/std pooled over all sensors' training observations.
    """
    train = _train(x, splits)
    n_obs = np.isfinite(train).sum(axis=1)
    if (n_obs == 0).any():
        raise ValueError(f"Sensors without training observations: {np.flatnonzero(n_obs == 0).tolist()}")
    if mode == "per_sensor":
        mean, std = np.nanmean(train, axis=1), np.nanstd(train, axis=1)
    elif mode == "global":
        mean = np.full(len(train), np.nanmean(train))
        std = np.full(len(train), np.nanstd(train))
    else:
        raise ValueError(f"Unknown scaler mode {mode!r}")
    std = np.where(std < MIN_STD, 1.0, std)
    return {
        "mode": mode,
        "fit_split": "train",
        "fit_range": list(splits["train"]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_obs": n_obs.tolist(),
    }


def _params(scaler, sensors, ndim):
    mean, std = np.asarray(scaler["mean"]), np.asarray(scaler["std"])
    if sensors is not None:
        mean, std = mean[sensors], std[sensors]
    shape = (-1,) + (1,) * (ndim - 1)
    return mean.reshape(shape), std.reshape(shape)


def transform(x, scaler, sensors=None):
    """Scale rows of x (first axis = sensor, or `sensors` gives each row's sensor). NaN stays NaN."""
    mean, std = _params(scaler, sensors, np.ndim(x))
    return (np.asarray(x, dtype=np.float64) - mean) / std


def inverse_transform(z, scaler, sensors=None):
    mean, std = _params(scaler, sensors, np.ndim(z))
    return np.asarray(z, dtype=np.float64) * std + mean


def fit_tod_median(x, splits, steps_per_day=STEPS_PER_DAY):
    """(sensor, slot) median of training observations; NaN where a slot was never observed."""
    start, end = splits["train"]
    train = _train(x, splits)
    slots = np.arange(start, end) % steps_per_day
    out = np.full((len(train), steps_per_day), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, handled later
        for slot in range(steps_per_day):
            out[:, slot] = np.nanmedian(train[:, slots == slot], axis=1)
    return out


def causal_fill(x, splits, tod_median, max_ffill=6, steps_per_day=STEPS_PER_DAY):
    """Fill missing *input* values without using any value from the future.

    Returns (filled, imputed) where imputed marks positions that were missing.
    """
    x = np.asarray(x, dtype=np.float64)
    observed = np.isfinite(x)
    filled = x.copy()
    train_mean = _nanmean_rows(_train(x, splits))
    if np.isnan(train_mean).any():
        raise ValueError("Cannot fill inputs for sensors without training observations")

    for start, end in splits.values():
        last = np.full(len(x), np.nan)
        age = np.zeros(len(x), dtype=np.int64)
        for t in range(start, end):
            obs = observed[:, t]
            last = np.where(obs, x[:, t], last)
            age = np.where(obs, 0, age + 1)
            use = ~obs & (age <= max_ffill) & np.isfinite(last)
            filled[use, t] = last[use]

    slots = np.arange(x.shape[1]) % steps_per_day
    still = np.isnan(filled)
    filled[still] = tod_median[:, slots][still]
    still = np.isnan(filled)
    filled[still] = np.broadcast_to(train_mean[:, None], x.shape)[still]
    return filled, ~observed


def _nanmean_rows(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)
