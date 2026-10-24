import numpy as np
import pytest

from aiori import preprocessing as pp
from aiori import splits

SPD = 4  # tiny "day" of 4 steps keeps arrays readable
SPLITS = {"train": (0, 8), "val": (8, 12), "test": (12, 16)}


def _series():
    x = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [0, 0, 10, np.nan, 0, 0, 10, 20, np.nan, 5, 5, 5, 5, 5, 5, 5],
    ], dtype=np.float64)
    return x


# --- scaler ---------------------------------------------------------------------

def test_scaler_uses_training_observations_only():
    x = _series()
    sc = pp.fit_scaler(x, SPLITS, mode="per_sensor")
    assert sc["mean"][0] == pytest.approx(4.5)
    assert sc["mean"][1] == pytest.approx(np.mean([0, 0, 10, 0, 0, 10, 20]))
    assert sc["n_obs"] == [8, 7]

    y = x.copy()
    y[:, 8:] = 1e6                      # perturb validation + test
    assert pp.fit_scaler(y, SPLITS, mode="per_sensor") == sc


def test_global_scaler_shares_parameters():
    sc = pp.fit_scaler(_series(), SPLITS, mode="global")
    assert sc["mean"][0] == sc["mean"][1] and sc["std"][0] == sc["std"][1]


def test_constant_training_series_gets_unit_std():
    x = _series()
    x[0, :8] = 3.0
    assert pp.fit_scaler(x, SPLITS)["std"][0] == 1.0


def test_no_training_observations_raises():
    x = _series()
    x[1, :8] = np.nan
    with pytest.raises(ValueError, match="training observations"):
        pp.fit_scaler(x, SPLITS)


def test_transform_round_trip_keeps_nan_and_zero():
    x = _series()
    sc = pp.fit_scaler(x, SPLITS)
    z = pp.transform(x, sc)
    assert np.isnan(z[1, 3]) and np.isnan(z[1, 8])
    back = pp.inverse_transform(z, sc)
    np.testing.assert_allclose(back, x, equal_nan=True)
    assert back[1, 0] == pytest.approx(0.0)

    # Window-shaped inverse with explicit sensor ids.
    w = z[[1, 0], 4:7]
    np.testing.assert_allclose(pp.inverse_transform(w, sc, sensors=np.array([1, 0])), x[[1, 0], 4:7])


# --- causal input imputation ----------------------------------------------------

def test_zeros_are_observed_not_filled():
    x = _series()
    tod = pp.fit_tod_median(x, SPLITS, steps_per_day=SPD)
    filled, imputed = pp.causal_fill(x, SPLITS, tod, max_ffill=2, steps_per_day=SPD)
    assert not imputed[1, [0, 1, 4, 5]].any()
    assert (filled[1, [0, 1, 4, 5]] == 0).all()


def test_short_gap_forward_filled_long_gap_uses_train_tod_median():
    x = np.full((1, 16), np.nan)
    x[0, :8] = [1, 2, 3, 4, 5, 6, 7, 8]          # train: two days of 4 steps
    x[0, 8] = 50.0
    tod = pp.fit_tod_median(x, SPLITS, steps_per_day=SPD)
    np.testing.assert_allclose(tod[0], [3, 4, 5, 6])
    filled, imputed = pp.causal_fill(x, SPLITS, tod, max_ffill=2, steps_per_day=SPD)
    # val: 50 observed, then two forward-filled steps, then tod median for slot 3.
    np.testing.assert_allclose(filled[0, 8:12], [50, 50, 50, 6])
    # test segment starts with NaN: forward fill does NOT reach back into val.
    np.testing.assert_allclose(filled[0, 12:16], [3, 4, 5, 6])
    assert imputed[0, 9:16].all() and not imputed[0, 8]


def test_forward_fill_does_not_cross_split_boundary():
    x = np.arange(16, dtype=np.float64)[None, :]
    x[0, 8] = np.nan                               # first val step missing
    tod = pp.fit_tod_median(x, SPLITS, steps_per_day=SPD)
    filled, _ = pp.causal_fill(x, SPLITS, tod, max_ffill=6, steps_per_day=SPD)
    assert filled[0, 8] == tod[0, 0] != x[0, 7]


def test_tod_median_ignores_val_and_test():
    x = _series()
    y = x.copy()
    y[:, 8:] = -999.0
    np.testing.assert_array_equal(pp.fit_tod_median(x, SPLITS, SPD), pp.fit_tod_median(y, SPLITS, SPD))


def test_fill_is_causal():
    """Filled value at t must not depend on raw values after t."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=(3, 16))
    x[rng.uniform(size=x.shape) < 0.3] = np.nan
    x[:, 0] = 1.0
    tod = pp.fit_tod_median(x, SPLITS, SPD)
    base, _ = pp.causal_fill(x, SPLITS, tod, max_ffill=3, steps_per_day=SPD)
    for t in range(8, 15):                         # perturb only the future (val/test)
        y = x.copy()
        y[:, t + 1:] = 1e9
        got, _ = pp.causal_fill(y, SPLITS, tod, max_ffill=3, steps_per_day=SPD)
        np.testing.assert_array_equal(got[:, : t + 1], base[:, : t + 1])


def test_fill_fallback_to_train_mean_when_slot_never_observed():
    x = np.arange(16, dtype=np.float64)[None, :]
    x[0, [3, 7]] = np.nan                          # slot 3 never observed in train
    x[0, 11] = np.nan
    tod = pp.fit_tod_median(x, SPLITS, SPD)
    assert np.isnan(tod[0, 3])
    filled, _ = pp.causal_fill(x, SPLITS, tod, max_ffill=0, steps_per_day=SPD)
    assert filled[0, 11] == pytest.approx(np.nanmean(x[0, :8]))
    assert np.isfinite(filled).all()


def test_split_config_from_real_helper_is_accepted():
    s = splits.make_splits(4 * SPD, fractions=(0.5, 0.25, 0.25), steps_per_day=SPD)
    assert pp.fit_scaler(_series()[:, :16], s)["fit_range"] == list(s["train"])
