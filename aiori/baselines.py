"""Reference forecasts on window indices (raw units, NumPy only).

Each forecast for window (s, t0) uses only raw values at steps < t0 or training statistics
(`tod_median`, fitted on the training split by preprocessing). Missing values are never
read as numbers: they fall back to the historical average.

    persistence          last observed raw value in [t0 - input_len, t0)
    seasonal_naive_daily raw[s, t0 + h - steps_per_day]   (always < t0 because h < horizon <= steps_per_day)
    historical_average   tod_median[s, (t0 + h) % steps_per_day]
"""

import warnings

import numpy as np

from aiori.config import STEPS_PER_DAY
from aiori.windows import HORIZON, INPUT_LEN


def _target_cols(index, horizon):
    return index[:, 1][:, None] + np.arange(horizon)


def historical_average(tod_median, index, horizon=HORIZON, steps_per_day=STEPS_PER_DAY):
    """Training time-of-day median; a slot never observed in training uses that sensor's mean median."""
    tod = np.asarray(tod_median, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_mean = np.nanmean(tod, axis=1)
    tod = np.where(np.isfinite(tod), tod, row_mean[:, None])
    if not np.isfinite(tod).all():
        raise ValueError("Sensor without any training time-of-day median")
    sensor = index[:, 0][:, None]
    return tod[sensor, _target_cols(index, horizon) % steps_per_day]


def persistence(flow_raw, tod_median, index, input_len=INPUT_LEN, horizon=HORIZON, steps_per_day=STEPS_PER_DAY):
    sensor, t0 = index[:, 0], index[:, 1]
    past = np.asarray(flow_raw, dtype=np.float64)[sensor[:, None], t0[:, None] + np.arange(-input_len, 0)]
    observed = np.isfinite(past)
    last = input_len - 1 - np.argmax(observed[:, ::-1], axis=1)       # latest observed input step
    value = past[np.arange(len(past)), last]
    has = observed.any(axis=1)
    out = np.repeat(value[:, None], horizon, axis=1)
    fallback = historical_average(tod_median, index, horizon, steps_per_day)
    out[~has] = fallback[~has]
    return out


def seasonal_naive_daily(flow_raw, tod_median, index, horizon=HORIZON, steps_per_day=STEPS_PER_DAY):
    if horizon > steps_per_day:
        raise ValueError("Daily seasonal-naive needs horizon <= steps_per_day")
    flow = np.asarray(flow_raw, dtype=np.float64)
    cols = _target_cols(index, horizon) - steps_per_day
    valid = cols >= 0
    value = flow[index[:, 0][:, None], np.where(valid, cols, 0)]
    use = valid & np.isfinite(value)
    return np.where(use, value, historical_average(tod_median, index, horizon, steps_per_day))


BASELINES = {
    "persistence": lambda run, idx: persistence(run.flow_raw, run.tod_median, idx,
                                                steps_per_day=run.steps_per_day),
    "seasonal_naive_daily": lambda run, idx: seasonal_naive_daily(run.flow_raw, run.tod_median, idx,
                                                                  steps_per_day=run.steps_per_day),
    "historical_average": lambda run, idx: historical_average(run.tod_median, idx, steps_per_day=run.steps_per_day),
}
