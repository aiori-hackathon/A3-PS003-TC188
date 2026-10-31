import numpy as np
import pytest

from aiori import metrics

Y_HAT = np.array([[1.0, 2.0], [3.0, 5.0], [0.0, -1.0]])
Y_TRUE = np.array([[2.0, np.nan], [3.0, 1.0], [1.0, np.nan]])
MASK = np.isfinite(Y_TRUE)


def test_pooled_matches_hand_calculation():
    # observed errors: -1, 0, 4, -1
    m = metrics.pooled(Y_HAT, Y_TRUE, MASK)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(6 / 4)
    assert m["rmse"] == pytest.approx(np.sqrt(18 / 4))


def test_masked_cells_ignored_even_with_nan_or_extreme_predictions():
    y_hat = Y_HAT.copy()
    y_hat[~MASK] = 1e9
    assert metrics.pooled(y_hat, Y_TRUE, MASK) == metrics.pooled(Y_HAT, Y_TRUE, MASK)


def test_nan_on_observed_cell_raises():
    y_hat = Y_HAT.copy()
    y_hat[0, 0] = np.nan
    with pytest.raises(ValueError):
        metrics.pooled(y_hat, Y_TRUE, MASK)


def test_all_masked_gives_nan_and_zero_n():
    m = metrics.pooled(Y_HAT, Y_TRUE, np.zeros_like(MASK))
    assert m["n"] == 0 and np.isnan(m["mae"]) and np.isnan(m["rmse"])


def test_by_horizon():
    h = metrics.by_horizon(Y_HAT, Y_TRUE, MASK)
    assert [x["horizon"] for x in h] == [1, 2]
    assert h[0]["n"] == 3 and h[0]["mae"] == pytest.approx(2 / 3)
    assert h[1]["n"] == 1 and h[1]["mae"] == pytest.approx(4.0)


def test_by_sensor_and_client_micro_macro():
    sensor_pos = np.array([0, 1, 1])
    client_of_sensor = np.array(["a", "a"])
    per_sensor = metrics.by_sensor(Y_HAT, Y_TRUE, MASK, sensor_pos)
    assert per_sensor[0] == {"mae": 1.0, "rmse": 1.0, "n": 1}
    assert per_sensor[1]["n"] == 3 and per_sensor[1]["mae"] == pytest.approx(5 / 3)
    c = metrics.by_client(Y_HAT, Y_TRUE, MASK, sensor_pos, client_of_sensor)["a"]
    assert c["mae"] == pytest.approx(6 / 4)                 # micro
    assert c["macro_mae"] == pytest.approx((1 + 5 / 3) / 2)  # mean of sensor MAEs
    assert c["n_sensors"] == 2
    assert metrics.sensor_macro(per_sensor)["mae"] == pytest.approx((1 + 5 / 3) / 2)


def test_metric_sums_reconstruct_pooled_across_parts():
    a1, s1, n1 = metrics.metric_sums(Y_HAT[:1], Y_TRUE[:1], MASK[:1])
    a2, s2, n2 = metrics.metric_sums(Y_HAT[1:], Y_TRUE[1:], MASK[1:])
    assert metrics.from_sums(a1 + a2, s1 + s2, n1 + n2) == metrics.pooled(Y_HAT, Y_TRUE, MASK)


def test_negative_rate_counts_observed_cells_only():
    r = metrics.negative_rate(Y_HAT, MASK)   # observed predictions 1, 3, 5, 0
    assert r == {"rate": 0.0, "n": 4}
    assert metrics.negative_rate(Y_HAT, np.ones_like(MASK))["rate"] == pytest.approx(1 / 6)
