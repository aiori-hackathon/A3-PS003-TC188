"""Phase 8 S5 pre-declared analysis (docs/phase8_200_sensor_technical_spec.md section 5).

Validation predictions only. Uncertainty comes from a stratified cluster bootstrap: whole sensors (or
whole freeway-direction groups) are resampled within each geographic client, so every sensor's
overlapping, autocorrelated forecast cells move together and are never treated as independent.
Arms are paired on identical cells; seeds are averaged per cell (seeds capture training randomness only).
"""

import numpy as np

CONTRASTS = {"C1": ("fedavg_geo", "centralised"), "C2": ("fedavg_geo", "local"),
             "C3": ("fedavg_iid", "fedavg_geo"), "C4": ("local", "centralised")}
BOOT_REPS = 2000
BOOT_SEED = 20261028
SESOI_FRACTION = 0.01
ALPHA = 0.05


def sesoi(centralised_mae, fraction=SESOI_FRACTION):
    """Smallest effect size of interest: `fraction` of the mean centralised pooled MAE."""
    return fraction * float(np.mean(centralised_mae))


def per_sensor_sums(values, mask, sensor_pos, n_sensors, key=None, n_keys=1):
    """Sum of `values` over observed cells per sensor, or per (sensor, key) when `key` (same shape) is given."""
    mask = np.asarray(mask, bool)
    rows = np.broadcast_to(np.asarray(sensor_pos)[:, None], mask.shape)[mask]
    v = np.asarray(values, np.float64)[mask]
    if key is None:
        return np.bincount(rows, weights=v, minlength=n_sensors)
    k = np.asarray(key)[mask]
    return np.bincount(rows * n_keys + k, weights=v, minlength=n_sensors * n_keys).reshape(n_sensors, n_keys)


def resample_totals(per_sensor, strata, clusters, reps=BOOT_REPS, seed=BOOT_SEED):
    """Stratified cluster bootstrap of totals.

    per_sensor: {name: (n_sensors,) sums}. Clusters (whole sensors or groups of sensors) are drawn with
    replacement within each stratum, as many as the stratum holds; every name uses the same draws, so
    contrasts are paired. Returns {name: (reps,) resampled totals}.
    """
    strata, clusters = np.asarray(strata), np.asarray(clusters)
    rng = np.random.default_rng(seed)
    out = {k: np.zeros(reps) for k in per_sensor}
    for s in sorted(np.unique(strata).tolist()):
        members = np.flatnonzero(strata == s)
        ids, inv = np.unique(clusters[members], return_inverse=True)
        idx = rng.integers(0, len(ids), size=(reps, len(ids)))
        for k, v in per_sensor.items():
            agg = np.bincount(inv, weights=np.asarray(v, np.float64)[members], minlength=len(ids))
            out[k] += agg[idx].sum(axis=1)
    return out


def interval_and_p(boot, reps=BOOT_REPS):
    """95% percentile interval and two-sided bootstrap p (2 x share on the far side of 0), floored at 1/reps."""
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return [float(lo), float(hi)], float(min(1.0, max(p, 1.0 / reps)))


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def verdict(ci, p_holm, diff, sesoi_value, alpha=ALPHA):
    """Pre-declared wording (spec section 5.3)."""
    excludes_zero = ci[0] > 0 or ci[1] < 0
    if excludes_zero and p_holm < alpha and abs(diff) >= sesoi_value:
        return "supported on January validation data"
    if excludes_zero:
        return "detectable but below the practical threshold"
    return "not distinguishable on this validation set"


def contrast(totals_a, totals_b, n, kind="mae"):
    """Pooled metric difference a - b from (bootstrap or point) totals of abs or squared errors."""
    if kind == "mae":
        return totals_a / n - totals_b / n
    return np.sqrt(totals_a / n) - np.sqrt(totals_b / n)
