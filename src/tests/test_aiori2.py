"""AIORI 2.0 Phase 2 preprocessing tests.

Synthetic tests run the whole pipeline on a tiny fake XTraffic tree (4 slots per day, 210 raw rows, 200 selected).
Real-data tests check the generated src/data/processed/aiori2_3yr outputs and are skipped when they are absent.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aiori import preprocessing
from aiori2 import core, pipeline
from aiori2.data import ThreeYearWindows, raw_metrics

SPD = 4
ROWS = 210


def _month_steps(y, m):
    nxt = (np.datetime64(f"{y}-{m:02d}") + np.timedelta64(1, "M")).astype("datetime64[D]")
    return int((nxt - np.datetime64(f"{y}-{m:02d}-01")).astype(int)) * SPD


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("aiori2_synth")
    rng = np.random.default_rng(1)
    node = rng.permutation(np.arange(1000, 1000 + ROWS))
    np.save(base / "node_order.npy", node)
    array_index = rng.choice(ROWS, 200, replace=False)               # deliberately unsorted
    clients = np.repeat([f"client_{k}" for k in range(4)], 50)[rng.permutation(200)]
    (base / "sel").mkdir()
    pd.DataFrame(dict(array_index=array_index, proposed_client=clients, type_group="mainline", flow_tertile=1,
                      selection_basis="x", station_id=node[array_index], District=7)).to_csv(base / "sel" / "sensor_selection.csv", index=False)
    (base / "protected").mkdir()
    (base / "protected" / "keep.txt").write_text("original")
    p = lambda i: array_index[i]                                      # raw row of selected position i
    t_glob = 0
    plants = {}
    for y in (2022, 2023, 2024):
        (base / f"year_{y}").mkdir()
        for m in range(1, 13):
            n = _month_steps(y, m)
            t = np.arange(t_glob, t_glob + n)
            x = np.empty((ROWS, n, 3))
            x[:, :, 0] = np.arange(ROWS)[:, None] * 10 + (t % SPD)[None] + 1 + (500 if y == 2022 else 0)
            x[:, :, 1], x[:, :, 2] = 0.1, 60.0
            if y == 2023 and m == 3:
                x[p(0), 5:9, 0] = np.nan                               # 2023 NaN gap
            if y == 2022 and m == 6:
                x[p(1), 10, :] = 0                                     # isolated all-three-zero reading
            if y == 2022 and m in (1, 2):                              # 2-step + 2-step all-zero run across Jan/Feb
                sl = slice(n - 2, n) if m == 1 else slice(0, 2)
                x[p(2), sl, :] = 0
            if y == 2022 and m in (3, 4):                              # 6-day outage across Mar/Apr (24 steps)
                sl = slice(n - 12, n) if m == 3 else slice(0, 12)
                x[p(5), sl, :] = 0
            if y == 2024 and m == 5:
                x[p(3), 7, 0], x[p(3), 7, 1] = 0, 0                    # zero flow with non-zero speed: valid
                x[p(4), 20:40, 0] = 123                                # constant non-zero run of 20 steps
            dt = np.float32 if y == 2023 else np.float64
            np.save(base / f"year_{y}" / f"{y}_p{m:02d}.npy", x.astype(dt))
            if y == 2024 and m == 5:
                plants["valid_zero_t"] = t_glob + 7
                plants["const"] = (t_glob + 20, t_glob + 40)
            if y == 2022 and m == 6:
                plants["isolated_t"] = t_glob + 10
            if y == 2022 and m == 2:
                plants["cross_t"] = (t_glob - 2, t_glob + 2)
            if y == 2022 and m == 4:
                plants["outage"] = (t_glob - 12, t_glob + 12)
            if y == 2023 and m == 3:
                plants["nan"] = (t_glob + 5, t_glob + 9)
            t_glob += n
    cfg = json.loads(core.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg.update(selection="sel/sensor_selection.csv", out_dir="out", protected_paths=["protected"], steps_per_day=SPD, step_minutes=360,
               quality={"long_run_steps": 8})
    cfg["policies"]["outage"]["all_zero_min_run_steps"] = 4
    (base / "cfg.json").write_text(json.dumps(cfg))
    c = core.load_config(base / "cfg.json", base_dir=base)
    for step in pipeline.STEPS.values():
        step(c)
    return SimpleNamespace(cfg=c, base=base, out=c["_out"], array_index=array_index, node=node, clients=clients, plants=plants)


def load(s, *parts):
    return np.load(s.out.joinpath(*parts))


# 1-4: identity, order, clients

def test_sensor_ids_order_and_clients_preserved(synth):
    sens = pd.read_csv(synth.out / "metadata" / "sensors.csv")
    assert len(sens) == 200 and sens.station_id.nunique() == 200
    assert (sens.array_index.to_numpy() == synth.array_index).all()
    assert (sens.station_id.to_numpy() == synth.node[synth.array_index]).all()
    assert (sens.client_geo.to_numpy() == synth.clients).all()
    assert sens.groupby("client_geo").size().tolist() == [50] * 4 and sens.groupby("client_iid").size().tolist() == [50] * 4
    flow = load(synth, "extracted", "flow_raw.npy")
    # each raw row has a distinctive level: position i must carry raw row array_index[i]
    for t, extra in ((1000, 500), (2000, 0)):                       # a 2022 step (+500 level) and a 2023 step
        assert np.allclose(flow[:, t], synth.array_index * 10 + (t % SPD) + 1 + extra)


def test_iid_assignment_is_the_study_partition(synth):
    from aiori.federated import iid_partition
    sens = pd.read_csv(synth.out / "metadata" / "sensors.csv")
    iid = iid_partition(SimpleNamespace(n_sensors=200), 20261027, 4)
    for name, pos in iid.items():
        assert (sens.client_iid.to_numpy()[pos] == name).all()


# 5-7: calendar

def test_calendar_steps_leap_year_and_weekdays():
    assert [core.year_steps(y) for y in (2022, 2023, 2024)] == [105120, 105120, 105408]
    cal = core.calendar([2022, 2023, 2024])
    ts, wd = cal["timestamp"], cal["weekday"]
    assert len(ts) == 315648 and str(ts[0]) == "2022-01-01T00:00" and str(ts[-1]) == "2024-12-31T23:55"
    at = lambda s: int((np.datetime64(s, "m") - ts[0]).astype(int) // 5)
    assert wd[at("2022-01-01T00:00")] == 5 and wd[at("2023-01-01T00:00")] == 6 and wd[at("2024-01-01T00:00")] == 0
    assert wd[at("2024-02-29T12:00")] == 3 and cal["month"][at("2024-02-29T12:00")] == 2 and cal["day"][at("2024-02-29T12:00")] == 29
    assert cal["slot"][at("2023-07-04T13:05")] == 13 * 12 + 1
    assert [str(core.us_spring_forward(y)) for y in (2022, 2023, 2024)] == ["2022-03-13", "2023-03-12", "2024-03-10"]
    assert cal["dst_nonexistent_hour"].sum() == 3 * 12 and cal["dst_nonexistent_hour"][at("2023-03-12T02:30")]


# 8-12: masks and flags

def test_2023_nan_is_masked_in_both_policies(synth):
    a, b = synth.plants["nan"]
    for pol in ("conservative", "outage"):
        v = load(synth, "masks", pol, "valid_target.npy")
        assert not v[0, a:b].any() and v[0, a - 1] and v[0, b]
    assert load(synth, "flags", "nonfinite_flow.npy")[0, a:b].all()


def test_isolated_all_zero_depends_on_policy(synth):
    t = synth.plants["isolated_t"]
    assert not load(synth, "masks", "conservative", "valid_target.npy")[1, t]
    assert load(synth, "masks", "outage", "valid_target.npy")[1, t]
    assert load(synth, "flags", "all_three_zero.npy")[1, t]
    flow = load(synth, "extracted", "flow_raw.npy")
    assert flow[1, t] == 0                                           # the raw value is preserved


def test_valid_zero_flow_is_kept(synth):
    t = synth.plants["valid_zero_t"]
    flow = load(synth, "extracted", "flow_raw.npy")
    assert flow[3, t] == 0
    for pol in ("conservative", "outage"):
        assert load(synth, "masks", pol, "valid_target.npy")[3, t]


def test_runs_crossing_month_boundaries_are_detected(synth):
    a, b = synth.plants["cross_t"]                                   # 2 + 2 steps across Jan/Feb: run of 4
    assert not load(synth, "masks", "outage", "valid_target.npy")[2, a:b].any()
    a, b = synth.plants["outage"]                                    # 12 + 12 steps across Mar/Apr
    assert load(synth, "flags", "suspected_outage_24h_conservative.npy")[5, a:b].all()
    rep = pd.read_csv(synth.out / "reports" / "quality_by_sensor_year.csv")
    r = rep[(rep.policy == "conservative") & (rep.year == 2022) & (rep.position == 5)].iloc[0]
    assert r.longest_suspected_outage_steps == 24


def test_constant_runs_are_flagged_not_removed(synth):
    a, b = synth.plants["const"]
    assert load(synth, "flags", "constant_run_24h.npy")[4, a:b].all()
    assert not load(synth, "flags", "constant_run_24h.npy")[4, a - 1]
    assert load(synth, "masks", "conservative", "valid_target.npy")[4, a:b].all()
    assert len(pd.read_csv(synth.out / "metadata" / "sensors.csv")) == 200


def test_run_length_helpers():
    m = np.array([[1, 1, 0, 1, 1, 1, 0]], bool)
    assert core.run_lengths(m).tolist() == [[2, 2, 0, 3, 3, 3, 0]]
    x = np.array([[5, 5, 5, 0, 0, 7, 7, np.nan, 7]], float)
    assert core.constant_runs(x).tolist() == [[3, 3, 3, 0, 0, 2, 2, 0, 0]]


# 13-17: scalers, splits, windows, causal filling

def _scalers(s, pol, exp):
    return json.loads((s.out / "experiments" / pol / exp / "scalers.json").read_text())


def test_scalers_never_see_validation_or_test(synth):
    off, splits, _ = pipeline.experiment_splits(synth.cfg, "B")
    flow = load(synth, "extracted", "flow_raw.npy")[:, off:].astype(np.float64)
    sens = pd.read_csv(synth.out / "metadata" / "sensors.csv")
    s1 = pipeline.fit_scalers(flow, splits, sens)
    flow2 = flow.copy()
    flow2[:, splits["val"][0]:] = 1e6                                # change every val/test value
    s2 = pipeline.fit_scalers(flow2, splits, sens)
    for scope in s1:
        assert s1[scope]["mean"] == s2[scope]["mean"] and s1[scope]["std"] == s2[scope]["std"]
    for exp in ("A", "B"):
        sc = _scalers(synth, "conservative", exp)
        _, sp, _ = pipeline.experiment_splits(synth.cfg, exp)
        assert sc["fit_range"] == [0, sp["val"][0]] and sc["train_dates"][1] == "2024-01-01"


def test_experiment_a_and_b_scalers_are_independent(synth):
    a, b = _scalers(synth, "conservative", "A"), _scalers(synth, "conservative", "B")
    for scope in ("per_sensor", "pooled", "client_geo", "client_iid"):
        assert a["scopes"][scope]["mean"] != b["scopes"][scope]["mean"]
    # A is exactly what fitting on 2023 alone gives
    off, splits, _ = pipeline.experiment_splits(synth.cfg, "A")
    flow = load(synth, "extracted", "flow_raw.npy")[:, off:].astype(np.float64)
    valid = load(synth, "masks", "conservative", "valid_target.npy")[:, off:]
    ref = preprocessing.fit_scaler(np.where(valid, flow, np.nan), splits, "per_sensor")
    assert np.allclose(a["scopes"]["per_sensor"]["mean"], ref["mean"])


def test_val_test_windows_identical_for_a_and_b_and_inside_splits(synth):
    for pol in ("conservative", "outage"):
        meta = {e: json.loads((synth.out / "experiments" / pol / e / "experiment.json").read_text()) for e in ("A", "B")}
        for split in ("val", "test"):
            w = {e: load(synth, "experiments", pol, e, "windows", f"{split}.npy").astype(np.int64) for e in ("A", "B")}
            absA = w["A"].copy(); absA[:, 1] += meta["A"]["timeline_offset"]
            absB = w["B"].copy(); absB[:, 1] += meta["B"]["timeline_offset"]
            assert np.array_equal(absA, absB) and len(absA) > 0
        for e in ("A", "B"):
            sp = meta[e]["splits_index"]
            for split in ("train", "val", "test"):
                w = load(synth, "experiments", pol, e, "windows", f"{split}.npy")
                s0, s1 = sp[split]
                assert (w[:, 1] >= s0).all() and (w[:, 1] + 12 <= s1).all() and (w[:, 1] >= 12).all()
    # experiment A never reaches back into 2022
    assert json.loads((synth.out / "experiments" / "conservative" / "A" / "experiment.json").read_text())["first_timestamp"].startswith("2023-01-01")


def test_causal_fill_never_uses_future_values():
    rng = np.random.default_rng(0)
    x = rng.uniform(10, 100, (3, 96))
    x[0, 40:50] = np.nan
    splits = {"train": (0, 48), "val": (48, 72), "test": (72, 96)}
    tod = preprocessing.fit_tod_median(x, splits, SPD)
    f1, imp = preprocessing.causal_fill(x, splits, tod, 6, SPD)
    for t in (41, 45, 55, 80):
        x2 = x.copy()
        x2[:, t:] = rng.uniform(-1e3, 1e3, x2[:, t:].shape)
        x2[0, 40:50] = np.where(np.arange(40, 50) < t, np.nan, x2[0, 40:50])
        f2, _ = preprocessing.causal_fill(x2, splits, tod, 6, SPD)
        assert np.array_equal(f1[:, :t], f2[:, :t])
    assert f1[0, 40] == x[0, 39] and imp[0, 40:50].all()


def test_pipeline_inputs_are_forward_filled_from_the_past(synth):
    off, _, _ = pipeline.experiment_splits(synth.cfg, "A")
    a, b = synth.plants["nan"]
    f = load(synth, "experiments", "conservative", "A", "filled_inputs.npy")
    flow = load(synth, "extracted", "flow_raw.npy")
    assert (f[0, a - off:b - off] == flow[0, a - 1]).all()
    assert load(synth, "experiments", "conservative", "A", "input_imputed.npy")[0, a - off:b - off].all()


# 18-19: evaluation masking and reopening outputs

def test_masked_targets_never_count(synth):
    ds = ThreeYearWindows(synth.out, "conservative", "A", "test", "client_geo")
    X, Y, M, s = ds.get(np.arange(min(len(ds), 500)))
    assert X.shape[1:] == (12, 1) and np.isfinite(X).all()
    assert np.array_equal(np.isfinite(Y), M)
    y_true = np.array([[10.0, 20, 30]]); mask = np.array([[True, True, False]])
    a = raw_metrics(np.array([[11.0, 18, 0]]), y_true, mask)
    b = raw_metrics(np.array([[11.0, 18, 1e9]]), y_true, mask)
    assert a == b and a["n"] == 2 and a["mae"] == 1.5 and abs(a["wape_pct"] - 10.0) < 1e-12
    z = (Y[:, :1] - ds.mean[s][:, None]) / ds.std[s][:, None]
    ok = np.isfinite(z[:, 0])
    assert np.allclose(ds.inverse(z, s)[ok], Y[:, :1][ok], atol=1e-3)


def test_outputs_reopen_and_verify(synth):
    res = pipeline.verify(synth.cfg)
    assert res["ok"], res["problems"]
    (synth.base / "protected" / "keep.txt").write_text("changed")
    try:
        assert not pipeline.verify(synth.cfg)["ok"]
    finally:
        (synth.base / "protected" / "keep.txt").write_text("original")


def test_extraction_refuses_lossy_float32_conversion(tmp_path, synth):
    x = np.load(synth.base / "year_2022" / "2022_p01.npy")
    bad = tmp_path / "base"
    import shutil
    shutil.copytree(synth.base, bad, ignore=shutil.ignore_patterns("out"))
    x[synth.array_index[0], 0, 0] = 0.1                              # a selected row; 0.1 is not exact in float32
    np.save(bad / "year_2022" / "2022_p01.npy", x)
    c = core.load_config(bad / "cfg.json", base_dir=bad)
    with pytest.raises(SystemExit, match="float32"):
        pipeline.extract(c)


# real data (skipped until `python -m aiori2 all` has run)

REAL = core.load_config()
HAS_REAL = (REAL["_out"] / "manifests" / "dgx_transfer_manifest.json").is_file()
real = pytest.mark.skipif(not HAS_REAL, reason="aiori2_3yr outputs not generated")


@real
def test_real_sensors_match_the_existing_study():
    sens = pd.read_csv(REAL["_out"] / "metadata" / "sensors.csv")
    sel = pd.read_csv(REAL["_base"] / REAL["selection"])
    assert (sens.array_index == sel.array_index).all() and (sens.station_id == sel.station_id).all()
    assert (sens.client_geo == sel.proposed_client).all() and sens.groupby("client_geo").size().tolist() == [50] * 4
    from aiori import dataset, federated
    d = REAL["_base"] / "data" / "processed" / "nine_month"
    run = dataset.load_run("nine_month", run_dir=d, selection=d / "sensor_selection.csv", splits=("val",))
    for name, pos in federated.geographic_partition(run).items():
        assert (sens.client_geo.to_numpy()[pos] == name).all()
    for name, pos in federated.iid_partition(run, 20261027).items():
        assert (sens.client_iid.to_numpy()[pos] == name).all()


@real
def test_real_shapes_and_steps():
    tl = json.loads((REAL["_out"] / "metadata" / "timeline.json").read_text())
    assert tl["n_steps"] == 315648 and tl["year_start_index"] == {"2022": 0, "2023": 105120, "2024": 210240}
    assert np.load(REAL["_out"] / "extracted" / "flow_raw.npy", mmap_mode="r").shape == (200, 315648)
    man = json.loads((REAL["_out"] / "manifests" / "extraction.json").read_text())
    assert len(man) == 36 and all(m["conversion"]["flow_exact"] for m in man.values())


@real
def test_real_outputs_verify_and_protected_artifacts_unchanged():
    res = pipeline.verify(REAL)
    assert res["ok"], res["problems"]
    assert res["protected_files"] > 0
