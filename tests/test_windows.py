import numpy as np
import pytest

from aiori import windows

SPLITS = {"train": (0, 40), "val": (40, 60), "test": (60, 100)}
L, H = 12, 12


def _mask(s=2, t=100):
    return np.ones((s, t), dtype=bool)


def test_window_counts_match_closed_form():
    m = _mask()
    # train: t0 in [12, 28] -> 17 per sensor; val: [40, 48] -> 9; test: [60, 88] -> 29
    assert len(windows.window_index(m, SPLITS, "train", L, H)) == 2 * 17
    assert len(windows.window_index(m, SPLITS, "val", L, H)) == 2 * 9
    assert len(windows.window_index(m, SPLITS, "test", L, H)) == 2 * 29


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_targets_stay_inside_their_split(split):
    idx = windows.window_index(_mask(), SPLITS, split, L, H)
    start, end = SPLITS[split]
    t0 = idx[:, 1]
    assert (t0 >= start).all() and (t0 + H <= end).all() and (t0 - L >= 0).all()


def test_val_inputs_may_use_past_train_context_but_not_future():
    idx = windows.window_index(_mask(), SPLITS, "val", L, H)
    assert idx[:, 1].min() == 40        # first val window reads inputs 28..39 (train)
    assert idx[:, 1].max() + H == 60    # last val target ends at the val/test boundary


def test_windows_without_observed_targets_dropped():
    m = _mask(1)
    m[0, 40:60] = False
    m[0, 50] = True
    idx = windows.window_index(m, SPLITS, "val", L, H)
    assert idx[:, 1].tolist() == list(range(40, 49))       # all contain step 50
    idx2 = windows.window_index(m, SPLITS, "val", L, H, min_observed_targets=2)
    assert len(idx2) == 0


def test_gather_shapes_masks_and_alignment():
    s, t = 2, 100
    x = np.arange(s * t, dtype=np.float64).reshape(s, t)
    imputed = np.zeros((s, t), dtype=bool)
    imputed[1, 30] = True
    y = x.copy()
    y[1, 45] = np.nan
    ymask = np.isfinite(y)
    idx = np.array([[1, 40], [0, 12]])
    X, Y, M = windows.gather_windows(x, imputed, y, ymask, idx, L, H)
    assert X.shape == (2, L, 2) and Y.shape == (2, H) and M.shape == (2, H)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    np.testing.assert_array_equal(X[0, :, 0], x[1, 28:40])
    assert X[0, 2, 1] == 0.0 and X[0, 0, 1] == 1.0          # observed flag = not imputed
    np.testing.assert_array_equal(Y[1], x[0, 12:24])
    # Missing target stays NaN (never zero) and is masked out.
    assert np.isnan(Y[0, 5]) and not M[0, 5] and M[0].sum() == H - 1


def test_gather_rejects_inconsistent_mask():
    x = np.zeros((1, 100))
    y = x.copy()
    y[0, 50] = np.nan
    with pytest.raises(ValueError, match="mask"):
        windows.gather_windows(x, np.zeros_like(x, bool), y, np.ones_like(x, bool),
                               np.array([[0, 45]]), L, H)
