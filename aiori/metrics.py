"""Masked forecast metrics in raw units (NumPy only).

All functions take predictions `y_hat` (N, H), truths `y_true` (N, H; NaN where missing) and a
target mask `mask` (N, H; True = observed). Only observed cells count, and every result
reports the number of cells used, `n`. Errors are computed with `np.where`, so NaN truths
never leak into sums.
"""

import numpy as np


def masked_errors(y_hat, y_true, mask):
    """e = y_hat - y_true on observed cells, 0.0 elsewhere (float64)."""
    mask = np.asarray(mask, dtype=bool)
    diff = np.asarray(y_hat, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    if not np.isfinite(diff[mask]).all():
        raise ValueError("Non-finite error on an observed cell")
    return np.where(mask, diff, 0.0)


def metric_sums(y_hat, y_true, mask, axis=None):
    """(sum |e|, sum e^2, n) - the additive pieces clients can report without predictions."""
    e = masked_errors(y_hat, y_true, mask)
    return np.abs(e).sum(axis=axis), (e ** 2).sum(axis=axis), np.asarray(mask, bool).sum(axis=axis)


def from_sums(abs_sum, sq_sum, n):
    """MAE / RMSE from sums; NaN when no cell is observed."""
    n = int(n)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {"mae": float(abs_sum / n), "rmse": float(np.sqrt(sq_sum / n)), "n": n}


def pooled(y_hat, y_true, mask):
    """Micro MAE / RMSE over all observed cells (primary metric)."""
    return from_sums(*metric_sums(y_hat, y_true, mask))


def by_horizon(y_hat, y_true, mask):
    """List of metrics for horizon steps 1..H (5..60 min)."""
    a, s, n = metric_sums(y_hat, y_true, mask, axis=0)
    return [dict(from_sums(a[h], s[h], n[h]), horizon=h + 1) for h in range(len(n))]


def by_group(y_hat, y_true, mask, groups):
    """{group: metrics} pooled over the windows whose `groups` value matches."""
    groups = np.asarray(groups)
    a, s, n = metric_sums(y_hat, y_true, mask, axis=1)
    return {g.item(): from_sums(a[groups == g].sum(), s[groups == g].sum(), n[groups == g].sum())
            for g in np.unique(groups)}


def by_sensor(y_hat, y_true, mask, sensor_pos):
    return by_group(y_hat, y_true, mask, sensor_pos)


def by_client(y_hat, y_true, mask, sensor_pos, client_of_sensor):
    """Per client: micro metrics over its cells plus `macro_mae` / `macro_rmse` (mean over its sensors)."""
    client_of_sensor = np.asarray(client_of_sensor)
    out = by_group(y_hat, y_true, mask, client_of_sensor[np.asarray(sensor_pos)])
    per_sensor = by_sensor(y_hat, y_true, mask, sensor_pos)
    for c in out:
        sensors = [s for s in per_sensor if client_of_sensor[s] == c and per_sensor[s]["n"] > 0]
        out[c]["macro_mae"] = float(np.mean([per_sensor[s]["mae"] for s in sensors])) if sensors else float("nan")
        out[c]["macro_rmse"] = float(np.mean([per_sensor[s]["rmse"] for s in sensors])) if sensors else float("nan")
        out[c]["n_sensors"] = len(sensors)
    return out


def sensor_macro(per_sensor):
    """Equal-weight mean of per-sensor MAE / RMSE (sensors with n = 0 skipped)."""
    vals = [m for m in per_sensor.values() if m["n"] > 0]
    if not vals:
        return {"mae": float("nan"), "rmse": float("nan"), "n_sensors": 0}
    return {"mae": float(np.mean([m["mae"] for m in vals])), "rmse": float(np.mean([m["rmse"] for m in vals])),
            "n_sensors": len(vals)}


def negative_rate(y_hat, mask):
    """Share of observed-cell predictions below zero (flows cannot be negative; not clipped)."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    return {"rate": float((np.asarray(y_hat)[mask] < 0).mean()) if n else float("nan"), "n": n}
