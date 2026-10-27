"""Optimizer-state ablation: reset_each_round (default) vs persist_per_client."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, federated, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402
from conftest import make_run  # noqa: E402

CFG = {"calendar": True, "hidden_size": 16, "lr": 1e-3, "batch_size": 16, "grad_clip": 1.0,
       "local_epochs": 1, "rounds_max": 3, "patience": 5}
PERSIST = dict(CFG, optimizer_state="persist_per_client")
RESET = dict(CFG, optimizer_state="reset_each_round")


@pytest.fixture(autouse=True)
def _torch_setup():
    training.configure_torch(2)


@pytest.fixture
def run4():
    return make_run(n_sensors=4, clients=["n", "n", "s", "s"])


def _eq(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _init(seed=0):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in ForecastGRU(4, 16).state_dict().items()}


def test_default_is_reset_and_unknown_value_rejected(run4):
    assert federated.Client("n", run4, [0, 1], CFG).optimizer_state == "reset_each_round"
    with pytest.raises(ValueError):
        federated.Client("n", run4, [0, 1], dict(CFG, optimizer_state="shared"))


def test_explicit_reset_reproduces_default_fedavg(run4):
    part = federated.geographic_partition(run4)
    a = federated.run_fedavg(run4, part, CFG, seed=4)
    b = federated.run_fedavg(run4, part, RESET, seed=4)
    assert _eq(a["state"], b["state"]) and a["byte_log"] == b["byte_log"]
    assert [h["val_mae"] for h in a["history"]] == [h["val_mae"] for h in b["history"]]


def test_reset_mode_keeps_no_optimizer_between_rounds(run4):
    c = federated.Client("n", run4, [0, 1], RESET)
    s0 = _init()
    a, _, _ = c.local_train(s0, 11, max_batches=3)
    b, _, _ = c.local_train(s0, 11, max_batches=3)
    assert _eq(a, b) and c._optimizer is None and c._local_model is None


def test_persist_matches_manual_single_adam_across_rounds(run4):
    c = federated.Client("n", run4, [0, 1], PERSIST)
    g0, g1 = _init(0), _init(1)                  # round 2 starts from a different "global" model
    out1, _, _ = c.local_train(g0, 11, max_batches=3)
    out2, _, st = c.local_train(g1, 12, max_batches=3)

    model = ForecastGRU(4, 16)
    model.load_state_dict(g0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = dataset.WindowDataset(c.run, "train", calendar=True)
    training.train_one_epoch(model, dataset.make_loader(ds, 16, True, 11), opt, 1.0, 3)
    assert _eq(out1, model.state_dict())
    model.load_state_dict(g1)                   # sync parameters, keep the same Adam
    training.train_one_epoch(model, dataset.make_loader(ds, 16, True, 12), opt, 1.0, 3)
    assert _eq(out2, model.state_dict())
    assert c.adam_step_count() == 6 and st[-1]["adam_step_total"] == 6


def test_first_round_identical_then_modes_diverge(run4):
    cp, cr = federated.Client("n", run4, [0, 1], PERSIST), federated.Client("n", run4, [0, 1], RESET)
    g0, g1 = _init(0), _init(1)
    assert _eq(cp.local_train(g0, 11, max_batches=3)[0], cr.local_train(g0, 11, max_batches=3)[0])
    assert not _eq(cp.local_train(g1, 12, max_batches=3)[0], cr.local_train(g1, 12, max_batches=3)[0])


def test_each_round_starts_from_the_global_parameters(run4, monkeypatch):
    seen = []
    real = training.train_one_epoch

    def spy(model, *a, **k):
        seen.append({n: p.detach().clone() for n, p in model.named_parameters()})
        return real(model, *a, **k)

    monkeypatch.setattr(training, "train_one_epoch", spy)
    c = federated.Client("n", run4, [0, 1], PERSIST)
    g0, g1 = _init(0), _init(1)
    c.local_train(g0, 11, max_batches=2)
    c.local_train(g1, 12, max_batches=2)
    assert _eq(seen[0], g0) and _eq(seen[1], g1)


def test_optimizer_state_is_private_per_client(run4):
    a, b = federated.Client("n", run4, [0, 1], PERSIST), federated.Client("s", run4, [2, 3], PERSIST)
    g = _init()
    a.local_train(g, 1, max_batches=2)
    b.local_train(g, 2, max_batches=2)
    assert a._optimizer is not b._optimizer and a._local_model is not b._local_model
    ptrs = lambda c: {t.data_ptr() for s in c._optimizer.state.values() for t in s.values() if t.dim()}  # noqa: E731
    assert not ptrs(a) & ptrs(b)
    server = federated.Server(g)
    with pytest.raises(TypeError):
        server.receive(1, "n", a._optimizer.state_dict(), 10)       # optimiser state cannot be uploaded
    assert set(vars(server)) == {"state", "byte_log", "_inbox"}


def test_persist_fedavg_aggregates_exactly_and_bytes_unchanged(run4, monkeypatch):
    captured = []
    real = federated.Server.aggregate_round

    def agg(self):
        inbox = [(n, {k: v.clone() for k, v in s.items()}, w) for n, s, w in self._inbox]
        out = real(self)
        captured.append((inbox, {k: v.clone() for k, v in self.state.items()}))
        return out

    monkeypatch.setattr(federated.Server, "aggregate_round", agg)
    part = federated.geographic_partition(run4)
    fed = federated.run_fedavg(run4, part, dict(PERSIST, rounds_max=2), seed=0)
    for inbox, state in captured:
        _, states, weights = zip(*sorted(inbox, key=lambda x: x[0]))
        assert _eq(federated.aggregate(list(states), list(weights)), state)
    ref = federated.run_fedavg(run4, part, dict(RESET, rounds_max=2), seed=0)
    assert [b["bytes"] for b in fed["byte_log"]] == [b["bytes"] for b in ref["byte_log"]]
    steps = [h["client_stats"][c][-1]["adam_step_total"] for h in fed["history"] for c in sorted(part)]
    per_round = [-(-federated.Client(c, run4, part[c], PERSIST).n_train // 16) for c in sorted(part)]
    assert steps == per_round + [2 * s for s in per_round]


def test_persist_deterministic_per_seed(run4):
    part = federated.geographic_partition(run4)
    f1 = federated.run_fedavg(run4, part, PERSIST, seed=1)
    f2 = federated.run_fedavg(run4, part, PERSIST, seed=1)
    f3 = federated.run_fedavg(run4, part, PERSIST, seed=2)
    assert _eq(f1["state"], f2["state"]) and [h["val_mae"] for h in f1["history"]] == [h["val_mae"] for h in f2["history"]]
    assert not torch.equal(f1["state"]["head.weight"], f3["state"]["head.weight"])


def test_single_client_persist_equals_continuous_centralised_training(run4):
    fed = federated.run_fedavg(run4, {"all": np.arange(4)}, dict(PERSIST, hidden_size=64, rounds_max=2,
                                                                  patience=5), seed=3)
    training.seed_everything(3)
    model = ForecastGRU(4, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = dataset.WindowDataset(run4, "train", calendar=True)
    states = []
    for r in (1, 2):
        training.train_one_epoch(model, dataset.make_loader(ds, 16, True, federated.client_seed(3, r, 0)), opt, 1.0)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    best = states[fed["best_round"] - 1]
    assert _eq(fed["state"], best)


def test_runner_refuses_to_overwrite_existing_results(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "experiments" / "run_phase7.py"
    spec = importlib.util.spec_from_file_location("run_phase7", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = path.parent / "configs" / "phase7_ablation_fedavg_geo_b1_persist.json"
    existing = tmp_path / "primary" / "fedavg_geo_b1_persist_seed0_smoke"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text("{}")
    monkeypatch.setattr(runner.dataset, "load_run", lambda *a, **k: pytest.fail("data loaded"))
    with pytest.raises(SystemExit, match="overwrite"):
        runner.main(["--config", str(cfg), "--smoke", "--out-dir", str(tmp_path)])
