"""Phase 8 E1: inference-only evaluator (scripts/phase8_test_eval.py) on a synthetic project only."""

import hashlib
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, evaluation, federated, metrics, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402
from test_select_geo200 import ROOT, _load_script, _synthetic_project  # noqa: E402

ARMS = ("centralised", "local", "fedavg_iid", "fedavg_geo")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def _run_cfg(arm):
    cfg = json.loads((ROOT / "experiments" / "configs" / f"phase8_{arm}_b1.json").read_text())
    return dict(cfg, hidden_size=8, threads=2)


def _freeze(root, ev):
    """(Re)write the freeze manifest and E1 record for the synthetic root; returns the CLI hash arguments."""
    runs = root / "results" / "phase8" / "geo200"
    data = root / "data" / "geo200"
    cm = json.loads((data / "checkpoint_manifest.json").read_text())
    fz = dict(runs={d.name: {p.name: sha(p) for p in sorted(d.iterdir())} for d in sorted(runs.iterdir())},
              geo200_artifacts=cm["artifact_sha256"], geo200_checkpoint_manifest_sha256=sha(data / "checkpoint_manifest.json"),
              sesoi_log_sha256=sha(root / "results" / "phase8" / "analysis" / "sesoi_log.json"),
              code_sha256={"aiori/frozen_code.py": sha(root / "aiori" / "frozen_code.py")},
              configs_sha256={"phase8_selection.json": sha(root / "experiments" / "configs" / "phase8_selection.json")})
    tp = root / "results" / "phase8" / "test_protocol"
    tp.mkdir(parents=True, exist_ok=True)
    (tp / "frozen_checkpoints.json").write_text(json.dumps(fz))
    proto, frz = sha(root / "docs" / "phase8_test_protocol.md"), sha(tp / "frozen_checkpoints.json")
    (tp / "e1_record.json").write_text(json.dumps(dict(script_sha256=sha(ROOT / "scripts" / "phase8_test_eval.py"),
                                                       protocol_sha256=proto, freeze_sha256=frz)))
    with np.load(data / "window_index.npz") as w:
        n_test = len(w["test"])
    return ["--protocol-sha256", proto, "--freeze-sha256", frz, "--root", str(root), "--data-dir", str(data),
            "--expected-test-windows", str(n_test)]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("p8test")
    prep = _load_script("prepare_geo200")
    assert prep.main(_synthetic_project(root)) == 0              # writes root/data/geo200 (16 sensors, 4 clients)
    data = root / "data" / "geo200"
    run = dataset.load_run("geo200", run_dir=data, selection=data / "sensor_selection.csv", splits=("train", "val"))
    training.configure_torch(2)
    for arm in ARMS:
        cfg = _run_cfg(arm)
        for seed in (0, 1, 2):
            d = root / "results" / "phase8" / "geo200" / f"{arm}_b1_geo200_seed{seed}"
            d.mkdir(parents=True)
            (d / f"phase8_{arm}_b1.json").write_text(json.dumps(cfg))
            (d / "metrics.json").write_text("{}")
            torch.manual_seed(100 * seed + ARMS.index(arm))
            if arm == "centralised":
                model = ForecastGRU(4, 8)
                torch.save(model.state_dict(), d / "best_model.pt")
                pred = training.predict(model, dataset.WindowDataset(run, "val", calendar=True))    # S4 phase-6 path
            else:
                part = (federated.iid_partition(run, cfg["partition_seed"]) if arm == "fedavg_iid"
                        else federated.geographic_partition(run))
                (d / "partition.json").write_text(json.dumps({k: v.tolist() for k, v in sorted(part.items())}))
                if arm == "local":
                    states = {c: ForecastGRU(4, 8).state_dict() for c in sorted(part)}
                    for c, st in states.items():
                        torch.save(st, d / f"best_model_{c}.pt")
                else:
                    g = ForecastGRU(4, 8).state_dict()
                    torch.save(g, d / "best_model_global.pt")
                    states = {c: g for c in part}
                clients = [federated.Client(c, run, p, cfg) for c, p in sorted(part.items())]   # S4 phase-7 path
                pred = federated.merge_predictions([cl.predict_val(states[cl.name]) for cl in clients])
            evaluation.write_prediction_record(d / f"predictions_{cfg['arm']}_{seed}_val.npz", pred, run, {})
    (root / "aiori").mkdir()
    (root / "aiori" / "frozen_code.py").write_text("# frozen\n")
    (root / "results" / "phase8" / "analysis").mkdir(parents=True)
    (root / "results" / "phase8" / "analysis" / "sesoi_log.json").write_text('{"sesoi_mae": 0.1638}')
    (root / "experiments" / "configs").mkdir(parents=True)
    (root / "experiments" / "configs" / "phase8_selection.json").write_text("{}")
    (root / "docs").mkdir()
    (root / "docs" / "phase8_test_protocol.md").write_text("# synthetic protocol\n")
    return root


@pytest.fixture
def proj(base, tmp_path):
    root = tmp_path / "p"
    shutil.copytree(base, root)
    return root, _freeze(root, None)


@pytest.fixture
def ev():
    return _load_script("phase8_test_eval")


def _spy_splits(monkeypatch, ev, seen):
    real = ev.dataset.load_run

    def spy(*a, **k):
        seen.append(tuple(k.get("splits") or ()))
        return real(*a, **k)
    monkeypatch.setattr(ev.dataset, "load_run", spy)


def test_precheck_reproduces_s4_predictions_exactly_and_never_loads_test(proj, ev, monkeypatch):
    root, argv = proj
    seen, built = [], []
    _spy_splits(monkeypatch, ev, seen)
    real_ds = ev.dataset.WindowDataset
    monkeypatch.setattr(ev.dataset, "WindowDataset", lambda run, split, *a, **k: built.append(split) or real_ds(run, split, *a, **k))
    assert ev.main(["--mode", "precheck", *argv]) == 0
    log = json.loads((root / "results/phase8/test_protocol/precheck_log.json").read_text())
    assert log["passed"] and len(log["runs"]) == 12 and all(r["identical"] for r in log["runs"].values())
    assert log["windows_loaded"] == ["val"] and seen == [("val",)] and set(built) == {"val"}
    assert sum(r["models"] for r in log["runs"].values()) == 3 + 12 + 3 + 3            # 21 checkpoints
    with pytest.raises(SystemExit, match="overwrite"):
        ev.main(["--mode", "precheck", *argv])


def test_swapped_local_client_checkpoints_fail_the_precheck(proj, ev):
    root, _ = proj
    d = root / "results/phase8/geo200/local_b1_geo200_seed1"
    a, b = sorted(d.glob("best_model_*.pt"))[:2]
    tmp = a.read_bytes()
    a.write_bytes(b.read_bytes())
    b.write_bytes(tmp)
    argv = _freeze(root, None)                           # a manifest that froze the wrong files
    assert ev.main(["--mode", "precheck", *argv]) == 1
    log = json.loads((root / "results/phase8/test_protocol/precheck_log.json").read_text())
    assert not log["passed"] and [t for t, r in log["runs"].items() if not r["identical"]] == ["local_b1_geo200_seed1"]


@pytest.mark.parametrize("tamper", ["protocol_arg", "freeze_arg", "checkpoint", "val_record", "code", "script_record", "artifact"])
def test_hash_gates_fail_closed_before_anything_is_written(proj, ev, tamper):
    root, argv = proj
    argv = list(argv)
    if tamper == "protocol_arg":
        argv[1] = "0" * 64
    elif tamper == "freeze_arg":
        argv[3] = "0" * 64
    elif tamper == "checkpoint":
        with open(root / "results/phase8/geo200/fedavg_geo_b1_geo200_seed0/best_model_global.pt", "ab") as f:
            f.write(b"\0")
    elif tamper == "val_record":
        p = next((root / "results/phase8/geo200/centralised_b1_geo200_seed2").glob("predictions_*"))
        p.write_bytes(p.read_bytes() + b"\0")
    elif tamper == "code":
        (root / "aiori/frozen_code.py").write_text("# changed\n")
    elif tamper == "artifact":
        with open(root / "data/geo200/pilot_preprocessed.npz", "ab") as f:
            f.write(b"\0")
    else:
        rec = root / "results/phase8/test_protocol/e1_record.json"
        rec.write_text(json.dumps(dict(json.loads(rec.read_text()), script_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="Refusing"):
        ev.main(["--mode", "precheck", *argv])
    assert not (root / "results/phase8/test_protocol/precheck_log.json").exists()
    assert not (root / "results/phase8/test").exists()


def test_test_mode_requires_a_passed_precheck(proj, ev):
    root, argv = proj
    with pytest.raises(SystemExit, match="no E2"):
        ev.main(["--mode", "test", *argv])
    log = root / "results/phase8/test_protocol/precheck_log.json"
    assert ev.main(["--mode", "precheck", *argv]) == 0
    log.write_text(json.dumps(dict(json.loads(log.read_text()), passed=False)))
    with pytest.raises(SystemExit, match="did not pass"):
        ev.main(["--mode", "test", *argv])
    assert not (root / "results/phase8/test").exists()


def test_test_pass_runs_once_inference_only_with_marker_first(proj, ev, monkeypatch):
    root, argv = proj
    assert ev.main(["--mode", "precheck", *argv]) == 0
    out = root / "results/phase8/test"
    seen = []
    real = ev.dataset.load_run

    def spy(*a, **k):
        assert (out / "RUN_STARTED").exists()          # marker written before any test window is loaded
        seen.append(tuple(k.get("splits") or ()))
        return real(*a, **k)
    monkeypatch.setattr(ev.dataset, "load_run", spy)
    monkeypatch.setattr(torch.optim.Adam, "__init__", lambda *a, **k: pytest.fail("optimiser created"))
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("backward called"))
    assert ev.main(["--mode", "test", *argv]) == 0
    assert seen == [("test",)]
    log = json.loads((out / "run_log.json").read_text())
    assert log["windows_loaded"] == ["test"] and len(log["runs"]) == 12 and (out / "RUN_COMPLETED").exists()
    assert set(json.loads((out / "reference_forecasts_test.json").read_text())) == {"persistence", "seasonal_naive_daily", "historical_average"}
    keys = set()
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        m = json.loads((d / "metrics_test.json").read_text())
        with np.load(next(d.glob("predictions_*_test.npz"))) as z:
            assert np.isfinite(z["y_hat"]).all()
            assert m["metrics"]["pooled"]["mae"] == pytest.approx(metrics.pooled(z["y_hat"].astype(float), z["y_true"].astype(float), z["mask"])["mae"], rel=1e-6)
            keys.add(hashlib.sha256(z["sensor_pos"].tobytes() + z["t0"].tobytes() + z["mask"].tobytes()).hexdigest())
        assert m["meta"]["split"] == "test" and m["meta"]["seed"] in (0, 1, 2)
    assert len(keys) == 1
    with pytest.raises(SystemExit, match="runs once"):
        ev.main(["--mode", "test", *argv])


def test_local_only_routes_each_client_model_to_its_own_sensors(proj, ev, monkeypatch):
    root, argv = proj
    run_dir = root / "results/phase8/geo200/local_b1_geo200_seed0"
    cfg, part, states = ev.load_run_spec(run_dir)
    assert set(states) == set(part) and len(states) == 4
    data = root / "data/geo200"
    run = dataset.load_run("geo200", run_dir=data, selection=data / "sensor_selection.csv", splits=("val",))
    calls = []
    real = ev.federated.subset_run
    monkeypatch.setattr(ev.federated, "subset_run", lambda r, pos, name=None: calls.append((name, sorted(pos))) or real(r, pos, name=name))
    pred, _ = ev.predict_split(run, "val", cfg, part, states)
    assert [(n.split(":")[1], p) for n, p in calls] == [(c, sorted(part[c].tolist())) for c in sorted(part)]
    for c, pos in part.items():                          # each client's windows come from that client's model only
        sub = real(run, np.sort(pos), name=c)
        alone = training.predict(ev._model(states[c], 4, cfg["hidden_size"]), dataset.WindowDataset(sub, "val", calendar=True))
        sel = np.isin(pred["sensor_pos"], pos)
        assert np.array_equal(pred["y_hat_z"][sel], alone["y_hat_z"])
        other = next(o for o in sorted(part) if o != c)
        wrong = training.predict(ev._model(states[other], 4, cfg["hidden_size"]), dataset.WindowDataset(sub, "val", calendar=True))
        assert not np.array_equal(pred["y_hat_z"][sel], wrong["y_hat_z"])


def test_parameter_change_during_prediction_is_detected(proj, ev, monkeypatch):
    root, _ = proj
    cfg, part, states = ev.load_run_spec(root / "results/phase8/geo200/centralised_b1_geo200_seed0")
    data = root / "data/geo200"
    run = dataset.load_run("geo200", run_dir=data, selection=data / "sensor_selection.csv", splits=("val",))
    real = ev.training.predict

    def tamper(model, ds, *a, **k):
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return real(model, ds, *a, **k)
    monkeypatch.setattr(ev.training, "predict", tamper)
    with pytest.raises(AssertionError, match="parameters changed"):
        ev.predict_split(run, "val", cfg, part, states)


def test_wrong_expected_test_window_count_stops(proj, ev):
    root, argv = proj
    assert ev.main(["--mode", "precheck", *argv]) == 0
    argv = argv[:-1] + ["1"]
    with pytest.raises(AssertionError, match="expected 1"):
        ev.main(["--mode", "test", *argv])
    assert not (root / "results/phase8/test/RUN_COMPLETED").exists()
