import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, federated, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402
from conftest import make_run  # noqa: E402

CFG = {"calendar": True, "hidden_size": 16, "lr": 1e-3, "batch_size": 16, "grad_clip": 1.0,
       "local_epochs": 1, "rounds_max": 2, "patience": 5, "max_epochs": 2}


@pytest.fixture
def run4():
    training.configure_torch(2)
    return make_run(n_sensors=4, clients=["n", "n", "s", "s"])


def test_partitions_are_disjoint_covers(run4):
    for part in (federated.geographic_partition(run4), federated.iid_partition(run4, 5, n_clients=2)):
        federated.check_partition(part, 4)
    with pytest.raises(ValueError):
        federated.check_partition({"a": np.array([0, 1]), "b": np.array([1, 2, 3])}, 4)


def test_client_holds_only_its_own_rows(run4):
    c = federated.Client("s", run4, [2, 3], CFG)
    r = c.run
    assert r.n_sensors == 2 and all(v is not run4 for v in vars(c).values())
    for k in ("inputs_scaled", "input_imputed", "targets_scaled", "observed_mask", "flow_raw", "tod_median"):
        assert np.array_equal(getattr(r, k), getattr(run4, k)[[2, 3]], equal_nan=k in ("targets_scaled", "flow_raw"))
        assert not np.shares_memory(getattr(r, k), getattr(run4, k))
    assert r.scaler["mean"] == [run4.scaler["mean"][2], run4.scaler["mean"][3]]
    for split in ("train", "val", "test"):
        full = run4.windows[split]
        back = np.stack([c.global_positions[r.windows[split][:, 0]], r.windows[split][:, 1]], 1)
        assert np.array_equal(back, full[np.isin(full[:, 0], [2, 3])])


def test_client_training_batches_contain_only_own_train_windows(run4, monkeypatch):
    calls = []
    real = dataset.WindowDataset.get_batch

    def spy(self, rows):
        b = real(self, rows)
        calls.append((self.run.name, self.split, torch.is_grad_enabled(), b["sensor_pos"].numpy()))
        return b

    monkeypatch.setattr(dataset.WindowDataset, "get_batch", spy)
    federated.run_fedavg(run4, federated.geographic_partition(run4), CFG, seed=0)
    assert {s for _, s, _, _ in calls} == {"train", "val"}
    assert all(split == "train" for _, split, grad, _ in calls if grad)
    for name, _, _, pos in calls:
        assert pos.max() < 2                                    # local positions of a 2-sensor client
        assert name.endswith((":n", ":s"))


def test_server_rejects_anything_but_states_ints_and_sums(run4):
    model = ForecastGRU(4, 16)
    server = federated.Server(model.state_dict())
    for bad in (dataset.WindowDataset(run4, "train"), np.zeros(3), {"x": np.zeros(3)},
                ForecastGRU(2, 16).state_dict()):
        with pytest.raises(TypeError):
            server.receive(1, "n", bad, 10)
    for bad_n in (0, -1, 2.5, True, "10"):
        with pytest.raises(TypeError):
            server.receive(1, "n", model.state_dict(), bad_n)
    with pytest.raises(TypeError):
        server.pool([{"abs": 1.0, "sq": 1.0, "n": 1, "abs_h": [1.0], "sq_h": [1.0], "n_h": [1], "y_hat": [0.0]}])
    assert set(vars(server)) == {"state", "byte_log", "_inbox"}


def test_validation_sums_carry_no_predictions_or_identities(run4):
    c = federated.Client("n", run4, [0, 1], CFG)
    sums = c.validation_sums(ForecastGRU(4, 16).state_dict())
    assert set(sums) == federated.SUM_KEYS
    assert all(isinstance(v, (int, float)) or (isinstance(v, list) and len(v) == 12) for v in sums.values())


def test_test_windows_never_read(run4, monkeypatch):
    real = dataset.WindowDataset.__init__
    splits = []

    def spy(self, run, split, *a, **k):
        splits.append(split)
        real(self, run, split, *a, **k)

    monkeypatch.setattr(dataset.WindowDataset, "__init__", spy)
    federated.run_fedavg(run4, federated.geographic_partition(run4), CFG, seed=0)
    federated.run_local_only(run4, federated.geographic_partition(run4), CFG, seed=0, max_batches=2)
    assert set(splits) == {"train", "val"}


def _runner():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "experiments" / "run_phase7.py"
    spec = importlib.util.spec_from_file_location("run_phase7", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path.parent / "configs"


@pytest.mark.parametrize("extra", [["--evaluate", "test"], ["--split", "test"], ["--final"]])
def test_runner_has_no_test_path(extra):
    runner, configs = _runner()
    with pytest.raises(SystemExit):
        runner.parse_args(["--config", str(configs / "phase7_fedavg_geo_b1.json"), *extra])


def test_runner_refuses_partial_participation(tmp_path, monkeypatch):
    import json
    runner, configs = _runner()
    cfg = json.loads((configs / "phase7_fedavg_geo_b1.json").read_text())
    cfg["client_fraction"] = 0.5
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(runner.dataset, "load_run", lambda *a, **k: pytest.fail("data loaded"))
    with pytest.raises(SystemExit, match="full participation"):
        runner.main(["--config", str(p)])
