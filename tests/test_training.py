import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, evaluation, training  # noqa: E402
from aiori.losses import masked_mse  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402


@pytest.fixture(autouse=True)
def _torch_setup():
    training.configure_torch(2)


def _state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_fully_masked_batch_takes_no_step_and_is_counted():
    torch.manual_seed(0)
    model = ForecastGRU(2)
    before = _state(model)
    batch = {"x": torch.randn(4, 12, 2), "y": torch.zeros(4, 12), "mask": torch.zeros(4, 12, dtype=torch.bool)}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = training.train_one_epoch(model, [batch], opt)
    assert stats["skipped_batches"] == 1 and stats["steps"] == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_tiny_batch_overfit(synthetic_run):
    training.seed_everything(0)
    ds = dataset.WindowDataset(synthetic_run, "train")
    batch = ds.get_batch(np.arange(0, len(ds), len(ds) // 16)[:16])
    model = ForecastGRU(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(400):
        stats = training.train_one_epoch(model, [batch], opt, grad_clip=1.0)
        first = first or stats["train_loss"]
    assert stats["train_loss"] < 0.01 and stats["train_loss"] < 0.01 * first


def _fit(run, seed, **kw):
    training.seed_everything(seed)
    model = ForecastGRU(2)
    out = training.fit(model, dataset.WindowDataset(run, "train"), dataset.WindowDataset(run, "val"), run,
                       seed=seed, batch_size=16, max_epochs=2, max_batches=5, **kw)
    return model, out


def test_same_seed_gives_identical_weights_and_metrics(synthetic_run):
    m1, f1 = _fit(synthetic_run, 7)
    m2, f2 = _fit(synthetic_run, 7)
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    assert [h["val_mae"] for h in f1["history"]] == [h["val_mae"] for h in f2["history"]]
    m3, _ = _fit(synthetic_run, 8)
    assert not torch.equal(m1.head.weight, m3.head.weight)


def test_early_stopping_restores_best_epoch(synthetic_run, monkeypatch):
    scripted = iter([5.0, 3.0, 4.0, 6.0, 1.0])
    snapshots = []
    real_predict = training.predict

    def predict(model, ds, *a, **k):
        snapshots.append(_state(model))
        return real_predict(model, ds, *a, **k)

    monkeypatch.setattr(training, "predict", predict)
    monkeypatch.setattr(training.evaluation, "evaluate_predictions",
                        lambda pred, run: {"pooled": {"mae": next(scripted), "rmse": 0.0, "n": 1}})
    training.seed_everything(0)
    model = ForecastGRU(2)
    out = training.fit(model, dataset.WindowDataset(synthetic_run, "train"),
                       dataset.WindowDataset(synthetic_run, "val"), synthetic_run,
                       seed=0, batch_size=16, max_epochs=10, patience=2, max_batches=2)
    assert out["best_epoch"] == 2 and len(out["history"]) == 4     # stops before the scripted 1.0
    assert out["early_stopped"]
    assert all(torch.equal(snapshots[1][k], v) for k, v in model.state_dict().items())


def test_inverse_scaling_on_mixed_sensor_batch(synthetic_run):
    sensor_pos = np.array([2, 0, 1, 0])
    z = np.random.default_rng(0).normal(size=(4, 12))
    mean, std = np.array(synthetic_run.scaler["mean"]), np.array(synthetic_run.scaler["std"])
    expected = z * std[sensor_pos][:, None] + mean[sensor_pos][:, None]
    assert np.allclose(evaluation.to_raw(z, synthetic_run, sensor_pos), expected)


def test_predicting_the_targets_scores_zero_and_record_round_trips(synthetic_run, tmp_path):
    ds = dataset.WindowDataset(synthetic_run, "val")
    b = ds.get_batch(np.arange(len(ds)))
    pred = {"y_hat_z": b["y"].numpy(), "mask": b["mask"].numpy(),
            "sensor_pos": b["sensor_pos"].numpy(), "t0": b["t0"].numpy()}
    res = evaluation.evaluate_predictions(pred, synthetic_run)
    assert res["pooled"]["mae"] < 1e-3 and res["pooled"]["n"] == int(pred["mask"].sum())
    assert res["pooled"]["n"] < pred["mask"].size                 # missing targets excluded
    path = tmp_path / "predictions_x_0_val.npz"
    evaluation.write_prediction_record(path, pred, synthetic_run, {"seed": 0})
    with np.load(path) as z:
        assert np.array_equal(np.isfinite(z["y_true"]), z["mask"])
        assert np.array_equal(z["array_index"], synthetic_run.array_index[pred["sensor_pos"]])
        assert json.loads(str(z["meta"])) == {"seed": 0}


def test_reference_forecasts_score_on_validation(synthetic_run):
    out = evaluation.evaluate_baselines(synthetic_run, "val")
    assert set(out) == {"persistence", "seasonal_naive_daily", "historical_average"}
    assert all(np.isfinite(v["pooled"]["mae"]) for v in out.values())


@pytest.mark.data
@pytest.mark.parametrize("name", ["primary", "missing_validation"])
def test_real_forward_backward_smoke(name):
    run = dataset.load_run(name)
    training.seed_everything(0)
    for split, calendar in (("train", False), ("val", True)):
        ds = dataset.WindowDataset(run, split, calendar=calendar)
        model = ForecastGRU(ds.n_features)
        batch = ds.get_batch(np.arange(len(ds)) if name == "missing_validation" and split == "val"
                             else np.arange(512))
        if name == "missing_validation" and split == "val":
            assert (~batch["mask"]).any(), "validation run must exercise masked targets"
        loss, n = masked_mse(model(batch["x"]), batch["y"], batch["mask"])
        loss.backward()
        assert torch.isfinite(loss) and n == int(batch["mask"].sum())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_non_finite_loss_raises_before_any_step():
    model = ForecastGRU(2)
    batch = {"x": torch.full((2, 12, 2), float("nan")), "y": torch.zeros(2, 12),
             "mask": torch.ones(2, 12, dtype=torch.bool)}
    with pytest.raises(FloatingPointError):
        training.train_one_epoch(model, [batch], torch.optim.Adam(model.parameters()))
