import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, federated, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402
from conftest import make_run  # noqa: E402

CFG = {"calendar": True, "hidden_size": 16, "lr": 1e-3, "batch_size": 16, "grad_clip": 1.0,
       "local_epochs": 1, "rounds_max": 3, "patience": 5, "max_epochs": 2}


@pytest.fixture(autouse=True)
def _torch_setup():
    training.configure_torch(2)


@pytest.fixture
def run4():
    return make_run(n_sensors=4, clients=["n", "n", "s", "s"])


def _states(k, seed=0):
    torch.manual_seed(seed)
    return [ForecastGRU(4, 8).state_dict() for _ in range(k)]


def test_aggregate_matches_numpy_weighted_mean():
    states, w = _states(3), [1, 2, 5]
    out = federated.aggregate(states, w)
    for k in out:
        ref = sum(wi * s[k].double().numpy() for wi, s in zip(np.array(w) / 8, states))
        assert out[k].dtype == torch.float32 and np.allclose(out[k].numpy(), ref, atol=1e-7)


def test_aggregate_equal_weights_is_mean_and_order_free():
    states = _states(4)
    a = federated.aggregate(states, [5, 5, 5, 5])
    b = federated.aggregate(states[::-1], [5, 5, 5, 5])
    for k in a:
        assert torch.equal(a[k], b[k])
        assert torch.allclose(a[k], torch.stack([s[k] for s in states]).mean(0), atol=1e-7)


def test_aggregating_identical_models_returns_the_model():
    s = _states(1)[0]
    out = federated.aggregate([s, s, s], [3, 1, 7])
    assert all(torch.equal(out[k], s[k]) for k in s)


def test_aggregate_rejects_bad_weights():
    with pytest.raises(ValueError):
        federated.aggregate(_states(2), [1, 0])


def test_single_client_fedavg_round_equals_one_centralised_epoch(run4):
    cfg = dict(CFG, rounds_max=1, hidden_size=64, batch_size=16)
    fed = federated.run_fedavg(run4, {"all": np.arange(4)}, cfg, seed=3)
    training.seed_everything(3)
    model = ForecastGRU(4, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = dataset.WindowDataset(run4, "train", calendar=True)
    training.train_one_epoch(model, dataset.make_loader(ds, 16, True, federated.client_seed(3, 1, 0)), opt, 1.0)
    assert all(torch.equal(fed["state"][k], v) for k, v in model.state_dict().items())


def test_steps_per_round_match_client_batches(run4):
    fed = federated.run_fedavg(run4, federated.geographic_partition(run4), CFG, seed=0)
    n = fed["n_k"]
    expected = sum(-(-v // CFG["batch_size"]) for v in n.values())
    assert all(h["steps"] == expected for h in fed["history"])
    assert sum(n.values()) == len(run4.windows["train"])        # one round = one centralised epoch of samples
    assert all(h["skipped_batches"] == 0 for h in fed["history"])


def test_optimizer_is_fresh_each_round(run4):
    c = federated.Client("n", run4, [0, 1], CFG)
    start = ForecastGRU(4, 16).state_dict()
    a, _, _ = c.local_train(start, shuffle_seed=11, max_batches=3)
    b, _, _ = c.local_train(start, shuffle_seed=11, max_batches=3)   # a carried Adam state would differ
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_reproducible_and_different_seed_differs(run4):
    part = federated.geographic_partition(run4)
    f1 = federated.run_fedavg(run4, part, CFG, seed=1)
    f2 = federated.run_fedavg(run4, part, CFG, seed=1)
    f3 = federated.run_fedavg(run4, part, CFG, seed=2)
    assert all(torch.equal(f1["state"][k], f2["state"][k]) for k in f1["state"])
    assert [h["val_mae"] for h in f1["history"]] == [h["val_mae"] for h in f2["history"]]
    assert f1["byte_log"] == f2["byte_log"]
    assert not torch.equal(f1["state"]["head.weight"], f3["state"]["head.weight"])


def test_client_shuffle_streams_are_independent(run4):
    c = federated.Client("n", run4, [0, 1], CFG)
    order = lambda s: np.concatenate([b["row"].numpy() for b in dataset.make_loader(c.train_ds, 16, True, s)])  # noqa: E731
    seeds = [federated.client_seed(0, 1, k) for k in range(4)]
    assert len(set(seeds)) == 4
    assert np.array_equal(order(seeds[0]), order(federated.client_seed(0, 1, 0)))
    assert federated.client_seed(0, 1, 0) != federated.client_seed(0, 2, 0) != federated.client_seed(1, 1, 0)


def test_early_stopping_restores_best_round(run4, monkeypatch):
    scripted = iter([5.0, 3.0, 4.0, 6.0, 1.0])
    real_pool = federated.Server.pool
    monkeypatch.setattr(federated.Server, "pool",
                        staticmethod(lambda sums: dict(real_pool(sums), mae=next(scripted))))
    snapshots = []
    real_agg = federated.Server.aggregate_round

    def agg(self):
        out = real_agg(self)
        snapshots.append({k: v.clone() for k, v in self.state.items()})
        return out

    monkeypatch.setattr(federated.Server, "aggregate_round", agg)
    fed = federated.run_fedavg(run4, federated.geographic_partition(run4), dict(CFG, rounds_max=10, patience=2),
                               seed=0)
    assert fed["best_round"] == 2 and len(fed["history"]) == 4 and fed["early_stopped"]
    assert all(torch.equal(fed["state"][k], snapshots[1][k]) for k in fed["state"])


def test_byte_log_counts_payloads(run4):
    fed = federated.run_fedavg(run4, federated.geographic_partition(run4), dict(CFG, rounds_max=2), seed=0)
    size = federated.payload_bytes(fed["state"])
    train = [b for b in fed["byte_log"] if b["purpose"] == "train"]
    assert len(train) == 2 * 2 * 2                                   # rounds x clients x {down, up}
    assert all(b["bytes"] == size for b in fed["byte_log"])
    assert all(h["train_bytes"] == 4 * size for h in fed["history"])
    assert size > federated.raw_bytes(fed["state"])


def test_iid_partition_is_balanced_deterministic_and_not_geographic():
    run = make_run(n_sensors=8, clients=list("aabbccdd"))
    p1, p2 = federated.iid_partition(run, 20261027), federated.iid_partition(run, 20261027)
    assert all(len(v) == 2 for v in p1.values())
    federated.check_partition(p1, 8)
    assert all(np.array_equal(p1[k], p2[k]) for k in p1)
    assert any(not np.array_equal(p1[k], federated.iid_partition(run, 1)[k]) for k in p1)
    geo = {tuple(v) for v in federated.geographic_partition(run).values()}
    assert {tuple(v) for v in p1.values()} != geo
    with pytest.raises(ValueError):
        federated.iid_partition(make_run(n_sensors=3), 0)


def test_local_only_and_pooled_sums_match_merged_records(run4):
    from aiori import evaluation
    part = federated.geographic_partition(run4)
    loc = federated.run_local_only(run4, part, CFG, seed=0)
    assert set(loc["clients"]) == set(part)
    pred = loc["predictions"]
    assert np.array_equal(np.stack([pred["sensor_pos"], pred["t0"]], 1), run4.windows["val"])
    fed = federated.run_fedavg(run4, part, dict(CFG, rounds_max=2), seed=0)
    from_records = evaluation.evaluate_predictions(fed["predictions"], run4)["pooled"]
    sums = fed["best_pooled_from_sums"]
    assert sums["n"] == from_records["n"]
    assert sums["mae"] == pytest.approx(from_records["mae"], rel=1e-9)
    assert sums["rmse"] == pytest.approx(from_records["rmse"], rel=1e-9)
    assert from_records["n"] < fed["predictions"]["mask"].size       # missing targets excluded
