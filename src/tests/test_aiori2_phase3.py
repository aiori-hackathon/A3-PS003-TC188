"""AIORI 2.0 Phase 3: four arms x two training histories (synthetic end-to-end fixture + real-data checks)."""

import json

import numpy as np
import pandas as pd
import pytest

from aiori2 import arms, core, pipeline, validate
from aiori2.data import ThreeYearWindows, raw_metrics
from test_aiori2 import synth  # noqa: F401  (module-scoped synthetic pipeline fixture)


def _streams(s, arm):
    return arms.streams(s.out, arm)


# partitions and arm structure

@pytest.mark.parametrize("arm, n_streams, scope, agg", [("centralised", 1, "pooled", False), ("local", 4, "client_geo", False),
                                                         ("fedavg_iid", 4, "client_iid", True), ("fedavg_geo", 4, "client_geo", True)])
def test_arm_streams_partition_all_200_sensors(synth, arm, n_streams, scope, agg):
    st = _streams(synth, arm)
    allpos = np.concatenate([x.sensors for x in st])
    assert len(st) == n_streams and np.array_equal(np.sort(allpos), np.arange(200))      # no omission, no overlap
    assert all(x.scaler_scope == scope for x in st) and arms.ARMS[arm]["aggregation"] is agg
    if n_streams == 4:
        assert all(len(x.sensors) == 50 for x in st)


def test_geo_streams_follow_the_selection_and_iid_the_study_seed(synth):
    from types import SimpleNamespace
    from aiori.federated import iid_partition
    geo = {x.name: x.sensors for x in _streams(synth, "fedavg_geo")}
    for c, pos in geo.items():
        assert (synth.clients[pos] == c).all()
    assert {x.name: x.sensors.tolist() for x in _streams(synth, "local")} == {k: v.tolist() for k, v in geo.items()}
    iid = iid_partition(SimpleNamespace(n_sensors=200), 20261027, 4)
    assert {x.name: x.sensors.tolist() for x in _streams(synth, "fedavg_iid")} == {k: v.tolist() for k, v in iid.items()}


@pytest.mark.parametrize("arm", ["local", "fedavg_iid", "fedavg_geo"])
def test_client_loaders_only_see_their_own_sensors(synth, arm):
    for name, ds in arms.loaders(synth.out, arm, "B", "train").items():
        own = next(x.sensors for x in _streams(synth, arm) if x.name == name)
        assert np.isin(np.asarray(ds.index[:, 0]), own).all() and len(ds) > 0
        X, Y, M, s = ds.get(arms.epoch_sample(len(ds), 64, 0, 0))
        assert np.isin(s, own).all()


def test_centralised_loader_sees_all_sensors_with_pooled_scaler(synth):
    (name, ds), = arms.loaders(synth.out, "centralised", "A", "train").items()
    assert name == "all" and set(np.unique(np.asarray(ds.index[:, 0]))) == set(range(200))
    assert len(set(ds.mean.tolist())) == 1 and ds.scope == "pooled"


def test_client_scalers_use_only_their_own_clients_training_data(synth):
    off, splits, _ = pipeline.experiment_splits(synth.cfg, "B")
    x = np.load(synth.out / "extracted" / "flow_raw.npy")[:, off:].astype(np.float64)
    sens = pd.read_csv(synth.out / "metadata" / "sensors.csv")
    base = pipeline.fit_scalers(x, splits, sens)
    for scope, col in (("client_geo", "client_geo"), ("client_iid", "client_iid")):
        for c, pos in pipeline.groups(sens, col).items():
            other = np.setdiff1d(np.arange(200), pos)
            x2 = x.copy()
            x2[other, :splits["train"][1]] *= 7                   # change every other client's training data
            s2 = pipeline.fit_scalers(x2, splits, sens)
            assert np.allclose(np.asarray(s2[scope]["mean"])[pos], np.asarray(base[scope]["mean"])[pos])
            assert np.allclose(np.asarray(s2[scope]["std"])[pos], np.asarray(base[scope]["std"])[pos])


# training histories and evaluation populations

def test_histories_are_separated_and_share_val_test(synth):
    ts = np.load(synth.out / "metadata" / "calendar" / "timestamp.npy")
    for arm in arms.ARMS:
        for split in ("val", "test"):
            la, lb = arms.loaders(synth.out, arm, "A", split), arms.loaders(synth.out, arm, "B", split)
            for c in la:
                a, b = la[c], lb[c]
                ta = a.target_timestamps(np.arange(len(a)), ts)
                tb = b.target_timestamps(np.arange(len(b)), ts)
                assert np.array_equal(ta, tb) and np.array_equal(np.asarray(a.index[:, 0]), np.asarray(b.index[:, 0]))
                _, Ya, Ma, _ = a.get(np.arange(min(len(a), 300)))
                _, Yb, Mb, _ = b.get(np.arange(min(len(b), 300)))
                assert np.array_equal(Ma, Mb) and np.array_equal(Ya, Yb, equal_nan=True)
        tra = arms.loaders(synth.out, arm, "A", "train")
        for ds in tra.values():
            first_input = ts[int(np.asarray(ds.index[:, 1]).min()) - 12 + ds.off]
            assert str(first_input) >= "2023-01-01"


# masks, metrics, sampling

def test_stream_scale_inverse_and_masked_metrics(synth):
    ds = arms.loaders(synth.out, "fedavg_iid", "A", "test")["iid_2"]
    X, Y, M, s = ds.get(np.arange(min(len(ds), 400)))
    assert X.shape == (len(s), 12, 1) and Y.shape == M.shape == (len(s), 12) and X.dtype == np.float32
    assert np.array_equal(np.isfinite(Y), M) and np.isfinite(X).all()
    back = ds.inverse(ds.scale(Y, s), s)
    assert np.allclose(back[M], Y[M], atol=1e-3)
    assert raw_metrics(Y, np.where(M, Y, 0), M)["mae"] == 0
    z = raw_metrics(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2), bool))
    assert z["mae"] == 0 and np.isnan(z["wape_pct"])            # zero denominator gives NaN, not a division error


def test_epoch_sampling_is_deterministic_and_without_replacement():
    a, b = arms.epoch_sample(1000, 100, 7, 0), arms.epoch_sample(1000, 100, 7, 0)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 100 and not np.array_equal(a, arms.epoch_sample(1000, 100, 7, 1))
    assert sorted(arms.epoch_sample(10, 50, 7, 0).tolist()) == list(range(10))


def test_quality_sensitivity_mask_is_evaluation_only(synth):
    valid_path = synth.out / "masks" / "conservative" / "valid_target.npy"
    before = core.sha256_file(valid_path)
    rep = validate.sensitivity(synth.cfg)
    keep = synth.out / "phase3" / "sensitivity" / "constant_run_24h" / "keep_mask.npy"
    assert core.sha256_file(valid_path) == before                 # primary mask untouched
    assert rep["by_year"]["2024"]["excluded_valid_targets"] >= 20 and rep["by_year"]["2024"]["sensors_affected"] >= 1
    a, b = synth.plants["const"]
    off, _, _ = pipeline.experiment_splits(synth.cfg, "B")
    plain = ThreeYearWindows(synth.out, "conservative", "B", "val", "pooled")
    sens = ThreeYearWindows(synth.out, "conservative", "B", "val", "pooled", eval_mask=keep)
    assert np.array_equal(np.asarray(plain.index), np.asarray(sens.index))     # same windows
    w = np.asarray(plain.index).astype(np.int64)
    hit = np.flatnonzero((w[:, 0] == 4) & (w[:, 1] + off <= a) & (w[:, 1] + off + 12 > a))
    if len(hit):                                                   # the planted run falls in the validation half-year
        _, _, Mp, _ = plain.get(hit[:1])
        _, _, Ms, _ = sens.get(hit[:1])
        assert Ms.sum() < Mp.sum()
    X1, _, _, _ = plain.get(np.arange(50))
    X2, _, _, _ = sens.get(np.arange(50))
    assert np.array_equal(X1, X2)                                  # inputs unchanged


def test_cpu_tensor_smoke(synth):
    torch = pytest.importorskip("torch")
    ds = arms.loaders(synth.out, "centralised", "B", "val")["all"]
    X, Y, M, s = ds.get(np.arange(min(len(ds), 128)))
    tx = torch.from_numpy(X)
    assert tx.dtype == torch.float32 and torch.isfinite(tx).all() and tx.shape[1:] == (12, 1)


def test_cuda_smoke(synth):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in this environment")
    ds = arms.loaders(synth.out, "centralised", "B", "val")["all"]
    X, _, _, _ = ds.get(np.arange(64))
    g = torch.from_numpy(X).to("cuda")
    assert g.is_cuda and torch.isfinite(g).all()


# real data

REAL = core.load_config()
HAS = (REAL["_out"] / "phase3" / "phase3_validation.json").is_file()
real = pytest.mark.skipif(not HAS, reason="Phase 3 validation not run")


@real
def test_real_phase3_validation_passed():
    rep = json.loads((REAL["_out"] / "phase3" / "phase3_validation.json").read_text())
    assert rep["ok"] and rep["problems"] == []


@real
@pytest.mark.parametrize("arm", list(arms.ARMS))
@pytest.mark.parametrize("history", arms.HISTORIES)
def test_real_experiment_manifests(arm, history):
    m = json.loads((core.SRC / "aiori2" / "configs" / "experiments" / f"{arm}_{history}.json").read_text())
    pos = np.concatenate([np.asarray(c["positions"]) for c in m["clients"].values()])
    assert np.array_equal(np.sort(pos), np.arange(200)) and len(m["station_ids"]) == 200
    assert m["scaler_scope"] == arms.ARMS[arm]["scaler"] and m["missing_value_policy"] == "conservative"
    assert m["val_dates"] == ["2024-01-01", "2024-07-01"] and m["test_dates"] == ["2024-07-01", "2025-01-01"]
    assert m["train_dates"][0] == ("2023-01-01" if history == "A" else "2022-01-01") and m["input_len"] == m["horizon"] == 12
