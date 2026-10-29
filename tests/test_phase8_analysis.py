"""Phase 8 S5 analysis helpers, on synthetic data only."""

import numpy as np
import pytest

from aiori import metrics, phase8_analysis as pa


def test_holm_known_values():
    assert np.allclose(pa.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert pa.holm([0.5, 0.9]) == [1.0, 1.0]


def test_sesoi_is_one_percent_of_mean_centralised_mae():
    assert pa.sesoi([16.0, 16.5, 17.0]) == pytest.approx(0.165)


@pytest.mark.parametrize("diff,ci,p,expected", [
    (0.30, [0.10, 0.50], 0.001, "supported on January validation data"),
    (0.10, [0.05, 0.15], 0.001, "detectable but below the practical threshold"),
    (0.30, [0.10, 0.50], 0.200, "detectable but below the practical threshold"),
    (0.30, [-0.10, 0.70], 0.400, "not distinguishable on this validation set"),
    (-0.30, [-0.50, -0.10], 0.010, "supported on January validation data")])
def test_verdict_wording(diff, ci, p, expected):
    assert pa.verdict(ci, p, diff, 0.164) == expected


def test_per_sensor_sums_match_pooled_metric_and_keys_partition_totals():
    rng = np.random.default_rng(0)
    y_hat, y = rng.normal(size=(50, 12)), rng.normal(size=(50, 12))
    mask = rng.random((50, 12)) > 0.1
    sp = rng.integers(0, 5, 50)
    e = np.abs(y_hat - y)
    a = pa.per_sensor_sums(e, mask, sp, 5)
    n = pa.per_sensor_sums(np.ones_like(e), mask, sp, 5)
    assert a.sum() / n.sum() == pytest.approx(metrics.pooled(y_hat, y, mask)["mae"])
    day = (np.arange(50)[:, None] + np.arange(12)) % 3
    by_day = pa.per_sensor_sums(e, mask, sp, 5, key=day, n_keys=3)
    assert np.allclose(by_day.sum(axis=1), a)


def test_bootstrap_is_seeded_stratified_and_paired():
    strata = np.repeat([0, 1], [3, 5])
    ones = {"n": np.ones(8), "x": np.arange(8.0)}
    t1 = pa.resample_totals(ones, strata, np.arange(8), reps=200, seed=1)
    t2 = pa.resample_totals(ones, strata, np.arange(8), reps=200, seed=1)
    assert all(np.array_equal(t1[k], t2[k]) for k in t1)
    assert (t1["n"] == 8).all()                       # each stratum keeps its size
    t3 = pa.resample_totals(ones, strata, np.arange(8), reps=200, seed=2)
    assert not np.array_equal(t1["x"], t3["x"])


def _synthetic(shift, n=80, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2, 5, n) * 100                  # per-sensor abs-error sums, heterogeneous
    a = base * (1 + rng.normal(0, 0.02, n)) + shift * 100
    return {"a": a, "b": base, "n": np.full(n, 100.0)}, np.repeat([0, 1, 2, 3], n // 4)


def test_known_difference_is_covered_and_null_is_not_rejected():
    sums, strata = _synthetic(0.5)
    t = pa.resample_totals(sums, strata, np.arange(80))
    diff = pa.contrast(t["a"], t["b"], t["n"])
    ci, p = pa.interval_and_p(diff)
    point = pa.contrast(sums["a"].sum(), sums["b"].sum(), sums["n"].sum())
    assert ci[0] < 0.5 < ci[1] and ci[0] > 0 and p == pytest.approx(1 / pa.BOOT_REPS)
    assert ci[0] <= point <= ci[1]
    null, strata = _synthetic(0.0, seed=3)
    null["a"] = null["b"] * (1 + np.random.default_rng(4).normal(0, 0.05, 80))
    t = pa.resample_totals(null, strata, np.arange(80))
    _, p0 = pa.interval_and_p(pa.contrast(t["a"], t["b"], t["n"]))
    assert p0 > 0.05


def test_cluster_bootstrap_is_wider_when_sensors_in_a_cluster_move_together():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(20), 4)            # 20 freeway groups x 4 sensors
    effect = rng.normal(0, 1, 20)[clusters] + rng.normal(0, 0.1, 80)
    sums = {"a": 100 + effect, "b": np.full(80, 100.0), "n": np.ones(80)}
    strata = np.zeros(80, int)
    width = lambda c: np.diff(pa.interval_and_p(pa.contrast(*(pa.resample_totals(sums, strata, c)[k] for k in "abn")))[0])[0]  # noqa: E731
    assert width(clusters) > 1.5 * width(np.arange(80))


def test_rmse_contrast_and_p_floor():
    assert pa.contrast(np.array(9.0), np.array(4.0), np.array(1.0), kind="rmse") == pytest.approx(1.0)
    _, p = pa.interval_and_p(np.full(100, 0.3), reps=100)
    assert p == pytest.approx(0.01)
    _, p = pa.interval_and_p(np.linspace(-1, 1, 101), reps=101)
    assert p == pytest.approx(1.0)
