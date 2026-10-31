import numpy as np
import pytest

from aiori import baselines

P = 8          # steps per day (small for tests)
L, H = 3, 2


def _data(s=2, days=4, seed=0):
    rng = np.random.default_rng(seed)
    flow = rng.uniform(10, 100, size=(s, days * P))
    tod = rng.uniform(10, 100, size=(s, P))
    return flow, tod


def test_persistence_uses_last_observed_input():
    flow, tod = _data()
    flow[0, 9] = np.nan                      # last input of t0 = 10 missing
    idx = np.array([[0, 10], [1, 10]])
    out = baselines.persistence(flow, tod, idx, L, H, P)
    assert np.array_equal(out[0], [flow[0, 8]] * H)
    assert np.array_equal(out[1], [flow[1, 9]] * H)


def test_persistence_falls_back_to_tod_median_when_no_input_observed():
    flow, tod = _data()
    flow[0, 7:10] = np.nan
    out = baselines.persistence(flow, tod, np.array([[0, 10]]), L, H, P)
    assert np.array_equal(out[0], tod[0, [10 % P, 11 % P]])


def test_seasonal_naive_reads_previous_day_and_falls_back():
    flow, tod = _data()
    flow[0, 10 + 1 - P] = np.nan
    out = baselines.seasonal_naive_daily(flow, tod, np.array([[0, 10], [1, 3]]), H, P)
    assert out[0, 0] == flow[0, 10 - P]
    assert out[0, 1] == tod[0, 11 % P]                 # missing -> median
    assert np.array_equal(out[1], tod[1, [3, 4]])      # before the start of the series -> median


def test_historical_average_fills_unobserved_slot_with_sensor_mean():
    _, tod = _data()
    tod[0, 2] = np.nan
    out = baselines.historical_average(tod, np.array([[0, 2]]), H, P)
    assert out[0, 0] == pytest.approx(np.nanmean(tod[0]))
    assert out[0, 1] == tod[0, 3]


@pytest.mark.parametrize("name", ["persistence", "seasonal_naive_daily"])
def test_no_access_at_or_after_t0(name):
    flow, tod = _data()
    idx = np.array([[0, 17], [1, 20]])
    fn = getattr(baselines, name)
    args = (L, H, P) if name == "persistence" else (H, P)
    before = fn(flow, tod, idx, *args)
    for s, t0 in idx:
        flow[s, t0:] = -999.0
    assert np.array_equal(before, fn(flow, tod, idx, *args))
