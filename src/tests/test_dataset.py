import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, windows  # noqa: E402
from conftest import P, SPLITS  # noqa: E402


def test_batch_shapes_dtypes_and_features(synthetic_run):
    ds = dataset.WindowDataset(synthetic_run, "train")
    b = ds.get_batch(np.arange(7))
    assert b["x"].shape == (7, 12, 2) and b["x"].dtype == torch.float32
    assert b["y"].shape == (7, 12) and b["y"].dtype == torch.float32
    assert b["mask"].shape == (7, 12) and b["mask"].dtype == torch.bool
    s, t0 = ds.index[:7, 0], ds.index[:7, 1]
    cols = t0[:, None] + np.arange(-12, 0)
    assert np.array_equal(b["x"][..., 0].numpy(), synthetic_run.inputs_scaled[s[:, None], cols])
    assert np.array_equal(b["x"][..., 1].numpy(), 1.0 - synthetic_run.input_imputed[s[:, None], cols])


def test_batch_equals_gather_windows_and_nan_only_replaced_where_masked(synthetic_run):
    ds = dataset.WindowDataset(synthetic_run, "val")
    rows = np.arange(len(ds))
    b = ds.get_batch(rows)
    r = synthetic_run
    X, Y, M = windows.gather_windows(r.inputs_scaled, r.input_imputed, r.targets_scaled, r.observed_mask, ds.index)
    assert (~M).any(), "synthetic val split must contain missing targets"
    assert np.array_equal(b["x"].numpy(), X)
    assert np.array_equal(b["mask"].numpy(), M)
    assert np.array_equal(b["y"].numpy()[M], Y[M])
    assert (b["y"].numpy()[~M] == 0).all() and np.isfinite(b["y"].numpy()).all()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_split_isolation(synthetic_run, split):
    ds = dataset.WindowDataset(synthetic_run, split)
    assert np.array_equal(ds.index, synthetic_run.windows[split])
    start, end = SPLITS[split]
    for b in dataset.make_loader(ds, 16):
        t0 = b["t0"].numpy()
        assert ((t0 >= start) & (t0 + 12 <= end)).all()


def test_client_filter(synthetic_run):
    pos = dataset.client_positions(synthetic_run)
    assert {k: v.tolist() for k, v in pos.items()} == {"a": [0, 1], "b": [2]}
    ds = dataset.WindowDataset(synthetic_run, "train", sensor_positions=pos["b"])
    assert set(ds.index[:, 0]) == {2}
    full = synthetic_run.windows["train"]
    assert np.array_equal(ds.index, full[full[:, 0] == 2])


def test_calendar_features_follow_input_steps(synthetic_run):
    ds = dataset.WindowDataset(synthetic_run, "train", calendar=True)
    b = ds.get_batch(np.array([0, 5]))
    assert b["x"].shape == (2, 12, 4)
    slot = (ds.index[[0, 5], 1][:, None] + np.arange(-12, 0)) % P
    assert np.allclose(b["x"][..., 2].numpy(), np.sin(2 * np.pi * slot / P), atol=1e-6)
    assert np.allclose(b["x"][..., 3].numpy(), np.cos(2 * np.pi * slot / P), atol=1e-6)


@pytest.mark.parametrize("shuffle", [False, True])
def test_loader_covers_each_row_once_with_matching_content(synthetic_run, shuffle):
    """The BatchSampler -> collate path yields exactly the intended windows, each once."""
    ds = dataset.WindowDataset(synthetic_run, "train")
    seen = []
    for b in dataset.make_loader(ds, 10, shuffle=shuffle, seed=3):
        rows = b["row"].numpy()
        assert len(rows) <= 10
        assert np.array_equal(torch.stack([b["sensor_pos"], b["t0"]], 1).numpy(), ds.index[rows])
        assert np.array_equal(b["x"].numpy(), ds.get_batch(rows)["x"].numpy())
        seen.append(rows)
    seen = np.concatenate(seen)
    assert np.array_equal(np.sort(seen), np.arange(len(ds)))
    assert (not shuffle) == np.array_equal(seen, np.arange(len(ds)))


def test_shuffle_order_is_deterministic_per_seed(synthetic_run):
    ds = dataset.WindowDataset(synthetic_run, "train")
    order = lambda seed: np.concatenate([b["row"].numpy() for b in dataset.make_loader(ds, 8, True, seed)])  # noqa: E731
    assert np.array_equal(order(1), order(1))
    assert not np.array_equal(order(1), order(2))
    with pytest.raises(ValueError):
        dataset.make_loader(ds, 8, shuffle=True)


def test_run_arrays_are_read_only(synthetic_run):
    ds = dataset.WindowDataset(synthetic_run, "train")
    with pytest.raises(ValueError):
        ds.index[0, 0] = 99


# ---- real artifacts (local only) ----

EXPECTED = {"primary": (40, {"train": 252520, "val": 34120, "test": 68680}),
            "missing_validation": (8, {"train": 50504, "val": 6824, "test": 13736})}


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_run_loads_with_expected_counts_and_finite_batches(name):
    run = dataset.load_run(name)
    n_sensors, counts = EXPECTED[name]
    assert run.n_sensors == n_sensors
    assert {k: len(v) for k, v in run.windows.items()} == counts
    for split in ("train", "val"):
        ds = dataset.WindowDataset(run, split, calendar=True)
        b = ds.get_batch(np.arange(0, len(ds), max(1, len(ds) // 2000)))
        assert torch.isfinite(b["x"]).all() and torch.isfinite(b["y"]).all()
        s, t0 = b["sensor_pos"].numpy(), b["t0"].numpy()
        nan = np.isnan(run.targets_scaled[s[:, None], t0[:, None] + np.arange(12)])
        assert np.array_equal(b["mask"].numpy(), ~nan)


@pytest.mark.data
def test_load_run_can_leave_test_windows_unloaded():
    run = dataset.load_run("primary", splits=("train", "val"))
    assert set(run.windows) == {"train", "val"}
    with pytest.raises(KeyError):
        dataset.WindowDataset(run, "test")
