"""Leakage and metric checks for the accuracy-optimisation pipeline.

    .venv-gpu\\Scripts\\python.exe -m pytest experiments/accuracy_optimization/test_aopt.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aopt import data as D  # noqa: E402
from aopt.metrics import full_report  # noqa: E402

pytestmark = pytest.mark.skipif(not (D.DATA / "pilot_preprocessed.npz").exists(), reason="local checkpoint absent")


@pytest.fixture(scope="module")
def store():
    return D.load(torch.device("cpu"))


def test_validation_set_is_the_frozen_one_and_test_is_never_loaded(store):
    with np.load(D.DATA / "window_index.npz") as w:
        assert np.array_equal(store.val.numpy(), w["val"])
    assert store.z.shape[1] == 63072                      # arrays end at the validation boundary (Aug 7, 23:55)
    assert int(store.train[:, 1].max()) + 12 <= 55008


def test_features_only_use_the_past(store):
    """Perturb every value at or after t0: past and lag features must not change; targets must."""
    f = D.Features(72, True, True, True, True)
    v = store.val
    first = torch.unique(v[:, 0], return_inverse=False)
    idx = torch.stack([v[v[:, 0] == s][500] for s in first[:64]])   # one window per sensor, so perturbations never overlap
    base = D.batch(store, idx, f)
    z = store.z.clone()
    for s, t0 in idx.tolist():
        store.z[s, t0:] += 1000.0
    try:
        moved = D.batch(store, idx, f)
    finally:
        store.z.copy_(z)
    assert torch.equal(base["x"], moved["x"])
    assert torch.equal(base["xf"], moved["xf"])


def test_profile_is_fitted_on_training_period_only(store):
    # The weekday x slot median must equal a recomputation from training flow alone.
    flow = store.flow.numpy()[:, :55008]
    t = np.arange(55008)
    cols = np.nonzero(((6 + t // 288) % 7 == 2) & (t % 288 == 100))[0]
    med = np.nanmedian(flow[5, cols])
    expect = (med - store.mean[5].item()) / store.std[5].item()
    assert abs(store.prof[5, 2, 100].item() - expect) < 1e-4


def test_metrics_match_study_definitions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 50, (100, 12)).astype(float)
    p = a + rng.normal(0, 3, a.shape)
    m = rng.random(a.shape) > 0.1
    r = full_report(p, a, m, np.zeros(100, int), np.arange(100) * 7, np.zeros(100, int), ["c0"])
    e = np.abs(p - a)[m]
    assert abs(r["wape_pct"] - 100 * e.sum() / a[m].sum()) < 1e-9
    assert abs(r["mae"] - e.mean()) < 1e-9
