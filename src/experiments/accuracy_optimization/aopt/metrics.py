"""Validation metrics with the frozen study definitions (raw units, observed cells only; mask = finite truth).

WAPE  = 100 * sum|actual - predicted| / sum|actual|
sMAPE = 100 * mean(2|a - p| / (|a| + |p|)), cells with a = p = 0 excluded
MAE / RMSE over the same cells. Identical to app/analysis/percentage_error.py and aiori.metrics.
"""

import numpy as np

SPD = 288
ORIGIN_WEEKDAY = 6  # 2023-01-01 was a Sunday (Monday = 0)


def wape(err, act):
    return float(np.abs(err).sum() / np.abs(act).sum())


def _block(e, a, p, m):
    den = np.abs(a) + np.abs(p)
    ok = m & (den > 0)
    sm = np.where(ok, 2 * np.abs(e) / np.where(den > 0, den, 1), 0.0)
    n = int(m.sum())
    return dict(wape_pct=100 * float(np.abs(e).sum() / np.abs(a).sum()), mae=float(np.abs(e).sum() / n),
                rmse=float(np.sqrt((e ** 2).sum() / n)), smape_pct=100 * float(sm.sum() / max(ok.sum(), 1)), n=n)


def full_report(y_hat, y_true, mask, sensor_pos, t0, client, client_names):
    """Pooled metrics plus breakdowns by horizon, client, hour of day, weekday/weekend and peak periods."""
    m = np.asarray(mask, bool)
    a = np.where(m, y_true, 0.0)
    p = np.where(m, y_hat, 0.0)
    e = p - a
    out = _block(e, a, p, m)
    out["by_horizon_wape_pct"] = [100 * float(np.abs(e[:, h]).sum() / a[:, h].sum()) for h in range(e.shape[1])]
    out["by_horizon_mae"] = [float(np.abs(e[:, h]).sum() / m[:, h].sum()) for h in range(e.shape[1])]
    client = np.asarray(client)
    out["by_client"] = {client_names[c]: _block(e[client == c], a[client == c], p[client == c], m[client == c])
                        for c in range(len(client_names))}
    # time of each target cell
    t = np.asarray(t0)[:, None] + np.arange(e.shape[1])
    hour = (t % SPD) // 12
    wd = (ORIGIN_WEEKDAY + t // SPD) % 7
    weekend = wd >= 5
    peak = ~weekend & (((hour >= 6) & (hour < 10)) | ((hour >= 15) & (hour < 19)))
    groups = {"weekday_peak": peak, "weekday_offpeak": ~weekend & ~peak, "weekend": weekend}
    out["by_period"] = {k: _block(e[g & m], a[g & m], p[g & m], m[g & m]) for k, g in groups.items()}
    out["by_hour_wape_pct"] = [100 * float(np.abs(e[hour == h]).sum() / a[hour == h].sum()) for h in range(24)]
    out["by_hour_share_of_abs_error"] = [float(np.abs(e[hour == h]).sum() / np.abs(e).sum()) for h in range(24)]
    return out
