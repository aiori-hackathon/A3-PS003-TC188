"""Phase 8 geo200 selection and checkpoint tooling, on synthetic tables and a synthetic raw array only."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiori import config, select_geo200 as sel
from aiori.flow_audit import _tie_key

ROOT = Path(__file__).resolve().parent.parent
DISTRICTS = [7, 4, 5, 12, 3, 6, 8, 10]
TYPES = ["Mainline", "HOV", "On Ramp", "Off Ramp", "Fwy-Fwy", "Coll/Dist", "Conventional Highway"]
TYPE_P = [0.55, 0.1, 0.12, 0.12, 0.05, 0.03, 0.03]


def make_tables(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    district = rng.choice(DISTRICTS, n)
    eligible = rng.random(n) > 0.15
    audit = pd.DataFrame({
        "array_index": np.arange(n), "station_id": 500000 + 7 * np.arange(n), "District": district,
        "proposed_client": pd.Series(district).map(config.CLIENT_BY_DISTRICT).to_numpy(),
        "missing_count": rng.integers(0, 60, n), "coverage_pct": rng.uniform(99, 100, n),
        "mean": rng.gamma(2.0, 100.0, n), "eligible": eligible,
        "duplicate_series": ~eligible & (rng.random(n) < 0.5),
        "month_mean": rng.gamma(2.0, 100.0, n), "month_coverage_pct": rng.uniform(90, 100, n),
        "eval_missing_count": rng.integers(0, 100, n),
        "exclusion_reason": np.where(eligible, "", "low_coverage"),
    })
    meta = pd.DataFrame({"station_id": audit["station_id"], "Type": rng.choice(TYPES, n, p=TYPE_P)})
    return audit, meta


def select(audit, meta, exclude=(), per_client=20):
    return sel.stratified_selection(sel.selection_frame(audit, meta), exclude, per_client)


@pytest.fixture
def tables():
    return make_tables()


# ---- allocation rules ----

def test_largest_remainder_known_values_and_tie_order():
    assert sel.largest_remainder({"a": 5, "b": 3, "c": 2}, 7)[0] == {"a": 4, "b": 2, "c": 1}
    assert sel.largest_remainder({"b": 1, "a": 1}, 1)[0] == {"a": 1, "b": 0}            # tie: name
    assert sel.largest_remainder({"a": 3, "b": 1, "c": 2}, 3)[0] == {"a": 2, "b": 0, "c": 1}  # tie: larger group
    alloc, order = sel.largest_remainder({"x": 2322, "y": 700, "z": 41}, 50)
    assert sum(alloc.values()) == 50 and order[0] in alloc


@pytest.mark.parametrize("k,expected", [(1, (0, 1, 0)), (2, (0, 1, 1)), (3, (1, 1, 1)), (4, (1, 2, 1)),
                                        (5, (1, 2, 2)), (27, (9, 9, 9))])
def test_tertile_allocation_remainder_mid_high_low(k, expected):
    assert tuple(sel.tertile_allocation(k)[t] for t in sel.TERTILES) == expected


def test_flow_tertiles_are_equal_thirds_with_deterministic_ties():
    mean = pd.Series([5.0] * 9, index=np.arange(10, 19))
    lab = sel.flow_tertiles(mean)
    assert lab.tolist() == ["low"] * 3 + ["mid"] * 3 + ["high"] * 3
    assert sel.flow_tertiles(pd.Series([3.0, 1.0])).tolist() == ["high", "low"]
    assert sel.flow_tertiles(pd.Series([3.0])).tolist() == ["mid"]


# ---- quotas, client assignment, reproducibility ----

def test_quotas_per_client_type_and_tertile(tables):
    audit, meta = tables
    chosen, summary = select(audit, meta)
    assert chosen["proposed_client"].value_counts().to_dict() == {c: 20 for c in set(config.CLIENT_BY_DISTRICT.values())}
    assert chosen["array_index"].is_unique and len(chosen) == 80
    cand = sel.candidates(sel.selection_frame(audit, meta))
    for client, info in summary["clients"].items():
        counts = cand.loc[cand["proposed_client"] == client, "type_group"].value_counts().to_dict()
        alloc = sel.largest_remainder(counts, 20)[0]
        got = chosen[chosen["proposed_client"] == client]
        assert got["type_group"].value_counts().reindex(sorted(alloc), fill_value=0).to_dict() == alloc
        if not info["fallbacks"]:
            for g, k in alloc.items():
                tert = got.loc[got["type_group"] == g, "flow_tertile"].value_counts()
                assert {t: int(tert.get(t, 0)) for t in sel.TERTILES} == sel.tertile_allocation(k)


def test_client_matches_district_map_and_mismatch_is_refused(tables):
    audit, meta = tables
    chosen, _ = select(audit, meta)
    assert (chosen["District"].map(config.CLIENT_BY_DISTRICT) == chosen["proposed_client"]).all()
    bad = audit.copy()
    bad.loc[bad.index[0], "proposed_client"] = "client_1_bay_area" if bad.at[0, "District"] != 4 else "client_0_la_ventura"
    with pytest.raises(ValueError, match="CLIENT_BY_DISTRICT"):
        select(bad, meta)


def test_deterministic_and_salt_dependent(tables, monkeypatch):
    audit, meta = tables
    a, sa = select(audit, meta)
    b, sb = select(audit.sample(frac=1, random_state=1).sort_values("array_index"), meta)
    pd.testing.assert_frame_equal(a, b)
    assert json.dumps(sa, default=int) == json.dumps(sb, default=int)
    monkeypatch.setattr(sel, "SALT", "another-salt")
    c, _ = select(audit, meta)
    assert set(c["array_index"]) != set(a["array_index"])


def test_draw_follows_salted_hash_within_strata(tables):
    audit, meta = tables
    chosen, _ = select(audit, meta)
    cand = sel.candidates(sel.selection_frame(audit, meta))
    for stratum, got in chosen[chosen["selection_basis"] == "stratified_hash"].groupby("stratum"):
        client, group, tert = stratum.split("/")
        c = cand[(cand["proposed_client"] == client) & (cand["type_group"] == group)].copy()
        c["flow_tertile"] = sel.flow_tertiles(c["mean"])
        pool = c[c["flow_tertile"] == tert].sort_values("_tie")["array_index"].tolist()
        assert got["array_index"].tolist() == pool[: len(got)]


# ---- leakage guards and missing data ----

def test_only_training_period_columns_influence_the_selection(tables):
    audit, meta = tables
    base, _ = select(audit, meta)
    rng = np.random.default_rng(9)
    shuffled = audit.copy()
    for col in ("month_mean", "month_coverage_pct", "eval_missing_count", "missing_count", "coverage_pct"):
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    pd.testing.assert_frame_equal(select(shuffled, meta)[0], base)


def test_forbidden_or_unlisted_columns_are_refused(tables):
    audit, meta = tables
    frame = sel.selection_frame(audit, meta)
    for col in ("month_mean", "eval_missing_count", "missing_count"):
        with pytest.raises(ValueError, match="accepts only"):
            sel.stratified_selection(frame.assign(**{col: audit[col]}), per_client=20)


def test_sensors_with_training_gaps_are_not_penalised(tables):
    audit, meta = tables
    gappy = audit.assign(missing_count=np.where(audit["array_index"] % 2 == 0, 60, 0))
    chosen, _ = select(gappy, meta)
    pd.testing.assert_frame_equal(chosen, select(audit, meta)[0])
    share = gappy.set_index("array_index").loc[chosen["array_index"], "missing_count"].gt(0).mean()
    assert 0.3 < share < 0.7


# ---- eligibility, duplicates, exclusions ----

def test_ineligible_and_exact_copies_never_selected(tables):
    audit, meta = tables
    chosen, _ = select(audit, meta)
    flags = audit.set_index("array_index").loc[chosen["array_index"]]
    assert flags["eligible"].all() and not flags["duplicate_series"].any()
    bad = audit.copy()
    bad.loc[bad["eligible"].idxmax(), "duplicate_series"] = True
    with pytest.raises(ValueError, match="A2"):
        select(bad, meta)


def test_pilot_sensors_excluded(tables):
    audit, meta = tables
    base, _ = select(audit, meta)
    exclude = base["array_index"].iloc[::3].tolist()
    chosen, summary = select(audit, meta, exclude)
    assert not chosen["array_index"].isin(exclude).any()
    assert summary["n_excluded_pilot"] == len(exclude) and len(chosen) == 80


def test_unmapped_type_fails_closed(tables):
    audit, meta = tables
    meta = meta.copy()
    meta.loc[audit["eligible"].idxmax(), "Type"] = "Unknown"
    with pytest.raises(ValueError, match="unmapped Type"):
        select(audit, meta)


# ---- fallbacks and District floor ----

def _client_tables(types, districts, means=None):
    n = len(types)
    audit = pd.DataFrame({"array_index": np.arange(n), "station_id": 900000 + np.arange(n), "District": districts,
                          "mean": means if means is not None else np.arange(1.0, n + 1),
                          "eligible": True, "duplicate_series": False})
    audit["proposed_client"] = audit["District"].map(config.CLIENT_BY_DISTRICT)
    return audit, pd.DataFrame({"station_id": audit["station_id"], "Type": types})


def test_short_tertile_falls_back_within_type_group():
    audit, meta = _client_tables(["Mainline"] * 4 + ["On Ramp"] * 6, [7] * 10)
    chosen, summary = select(audit, meta, per_client=10)
    assert len(chosen) == 10
    assert any(f["rule"] == "fallback_tertile" and f["type_group"] == "mainline" for f in summary["fallbacks"])
    assert (chosen["selection_basis"] == "fallback_tertile").sum() == 1


def test_exhausted_type_group_falls_back_to_next_group(monkeypatch):
    audit, meta = _client_tables(["Mainline"] * 12 + ["HOV"] * 3, [7] * 15)
    real = sel.largest_remainder
    monkeypatch.setattr(sel, "largest_remainder",
                        lambda counts, total: ({"hov": 5, "mainline": total - 5}, real(counts, total)[1]))
    chosen, summary = select(audit, meta, per_client=10)
    assert len(chosen) == 10 and (chosen["type_group"] == "hov").sum() == 3
    assert any(f["rule"] == "fallback_type" and f["to_type_group"] == "mainline" for f in summary["fallbacks"])


def test_district_floor_replaces_last_draw_of_largest_stratum():
    n = 60
    ties = [_tie_key(i, sel.SALT) for i in range(n)]
    last = int(np.argmax(ties))                          # hashed last: never drawn without the floor
    districts = [12] * n
    districts[last] = 5
    audit, meta = _client_tables(["Mainline"] * n, districts, means=np.full(n, 10.0))
    chosen, summary = select(audit, meta, per_client=9)
    assert len(chosen) == 9 and (chosen["District"] == 5).sum() == 1
    assert chosen.loc[chosen["District"] == 5, "selection_basis"].item() == "district_floor"
    assert [f["rule"] for f in summary["fallbacks"]] == ["district_floor"]


# ---- input hashes, config, output folder ----

def test_input_hash_check_fails_closed(tmp_path):
    f = tmp_path / "audit.csv"
    f.write_text("a\n1\n")
    good = {"audit.csv": sel.sha256_file(f)}
    assert sel.check_input_hashes({"audit.csv": f}, good) == good
    with pytest.raises(ValueError, match="changed"):
        sel.check_input_hashes({"audit.csv": f}, {"audit.csv": "0" * 64})
    with pytest.raises(ValueError, match="no recorded hash"):
        sel.check_input_hashes({"audit.csv": f, "x.csv": f}, good)


def test_selection_config_is_valid_and_validated(tmp_path):
    cfg = sel.load_config(ROOT / "experiments" / "configs" / "phase8_selection.json")
    assert cfg["per_client"] == 50 and cfg["out_dir"] == "data/processed/geo200"
    assert len(cfg["input_sha256"]) == 4 and all(len(h) == 64 for h in cfg["input_sha256"].values())
    for change in ({"salt": "x"}, {"per_client": 70}, {"input_sha256": {}}):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(dict(cfg, **change)))
        with pytest.raises(ValueError):
            sel.load_config(p)


def test_outputs_never_overwrite(tmp_path, tables):
    audit, meta = tables
    chosen, summary = select(audit, meta)
    manifest = sel.build_manifest(audit, chosen)
    out = sel.write_outputs(tmp_path / "geo", manifest, summary)
    assert {p.name for p in out.iterdir()} == {"sensor_selection.csv", "selection_summary.json", "selection_report.md"}
    with pytest.raises(FileExistsError):
        sel.write_outputs(out, manifest, summary)


def test_manifest_keeps_pilot_columns_and_selection_order(tables):
    audit, meta = tables
    chosen, _ = select(audit, meta)
    manifest = sel.build_manifest(audit, chosen)
    assert list(manifest.columns) == list(audit.columns) + list(sel.MANIFEST_EXTRA)
    assert manifest["array_index"].tolist() == chosen["array_index"].tolist()


def test_near_duplicate_screen_reports_only_high_correlation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    flow = np.stack([a, 2 * a + 1, rng.normal(size=500), np.full(500, 3.0)])
    flow[1, 10] = np.nan
    assert [p[:2] for p in sel.near_duplicate_pairs(flow)] == [(0, 1)]


# ---- stage wrapper end to end on a synthetic raw array ----

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_project(root, n=80, t=8928, perturb_eval=False):
    rng = np.random.default_rng(3)
    (root / "year_2023").mkdir(parents=True)
    steps = np.arange(t)
    flow = 100 + 60 * np.sin(2 * np.pi * steps / 288)[None] * rng.uniform(0.3, 2, (n, 1)) + rng.normal(0, 5, (n, t))
    flow[5, 100:104] = np.nan                         # training gap
    flow[7, 6400:6403] = np.nan                       # missing validation targets
    if perturb_eval:
        flow[:, 6336:] += 1000.0                      # val/test values must not affect selection or scaler
    arr = np.zeros((n, t, 3), np.float32)
    arr[:, :, 0] = flow
    np.save(root / "year_2023" / "2023_p01.npy", arr)
    np.save(root / "year_2023" / "2023_p02.npy", np.zeros(4, np.float32))   # must never be opened
    np.save(root / "node_order.npy", 500000 + 7 * np.arange(n))
    (root / "sensor_meta_feature.csv").write_text("station_id,Type\n")    # fingerprinted, never read
    audit, meta = make_tables(n, seed=4)
    audit["eligible"], audit["duplicate_series"], audit["exclusion_reason"] = True, False, ""
    audit["mean"] = np.nanmean(flow[:, :6336], axis=1)
    audit["District"] = np.resize([7, 4, 12, 8], n)       # one District per client: the floor does not bind
    audit["proposed_client"] = audit["District"].map(config.CLIENT_BY_DISTRICT)
    d = root / "data"
    d.mkdir()
    audit.to_csv(d / "audit.csv", index=False)
    meta.to_csv(d / "meta.csv", index=False)
    pd.DataFrame({"array_index": [0, 1]}).to_csv(d / "pilot.csv", index=False)
    names = {"audit": "data/audit.csv", "metadata": "data/meta.csv", "pilot_manifests": ["data/pilot.csv"]}
    cfg = {"salt": sel.SALT, "per_client": 4, "inputs": names, "out_dir": "data/geo200",
           "input_sha256": {p: sel.sha256_file(root / p) for p in ["data/audit.csv", "data/meta.csv", "data/pilot.csv"]},
           "prepare_pilot": {"min_sensors": 10, "max_sensors": 20, "run_label": "synthetic geo200"}}
    (root / "sel.json").write_text(json.dumps(cfg))
    return ["--config", str(root / "sel.json"), "--root", str(root), "--array", str(root / "year_2023" / "2023_p01.npy"),
            "--node-order", str(root / "node_order.npy"), "--raw-metadata", str(root / "sensor_meta_feature.csv"),
            "--no-shape-check"]


def test_stage_wrapper_end_to_end_and_split_isolation(tmp_path):
    prep = _load_script("prepare_geo200")
    argv = _synthetic_project(tmp_path / "a")
    assert prep.main(argv) == 0
    out = tmp_path / "a" / "data" / "geo200"
    rec = json.loads((out / "checkpoint_manifest.json").read_text())
    assert rec["raw_arrays_opened"] == ["2023_p01.npy"]
    assert rec["raw_fingerprints_before"] == rec["raw_fingerprints_after"]
    assert set(rec["raw_fingerprints_before"]) == {"2023_p01.npy", "2023_p02.npy", "node_order.npy",
                                                     "sensor_meta_feature.csv"}
    assert {"sensor_selection.csv", "pilot_preprocessed.npz", "window_index.npz", "split_config.json",
            "scaler_parameters.json", "preprocessing_report.md"} <= set(rec["artifact_sha256"])
    counts = rec["counts"]
    assert counts["sensors_per_client"] == {c: 4 for c in set(config.CLIENT_BY_DISTRICT.values())}
    assert "masked_target_cells" not in counts["splits"]["test"]
    manifest = pd.read_csv(out / "sensor_selection.csv")
    assert not manifest["array_index"].isin([0, 1]).any()

    bounds = json.loads((out / "split_config.json").read_text())["boundaries"]
    with np.load(out / "window_index.npz") as w:
        for name, (start, end) in bounds.items():
            t0 = w[name][:, 1]
            assert ((t0 >= start) & (t0 + 12 <= end) & (t0 - 12 >= 0)).all()
    scaler = json.loads((out / "scaler_parameters.json").read_text())
    assert scaler["fit_range"] == bounds["train"]

    argv_b = _synthetic_project(tmp_path / "b", perturb_eval=True)
    assert prep.main(argv_b) == 0
    out_b = tmp_path / "b" / "data" / "geo200"
    assert pd.read_csv(out_b / "sensor_selection.csv")["array_index"].tolist() == manifest["array_index"].tolist()
    scaler_b = json.loads((out_b / "scaler_parameters.json").read_text())
    assert scaler_b["mean"] == scaler["mean"] and scaler_b["std"] == scaler["std"]
    with np.load(out / "pilot_preprocessed.npz") as d, np.load(out_b / "pilot_preprocessed.npz") as e:
        a, b = bounds["train"]
        assert np.array_equal(d["inputs_scaled"][:, a:b], e["inputs_scaled"][:, a:b])
        assert np.array_equal(d["tod_median"], e["tod_median"], equal_nan=True)


def test_stage_wrapper_refuses_non_empty_output_before_reading(tmp_path, monkeypatch):
    prep = _load_script("prepare_geo200")
    argv = _synthetic_project(tmp_path)
    (tmp_path / "data" / "geo200").mkdir()
    (tmp_path / "data" / "geo200" / "pilot_preprocessed.npz").write_bytes(b"x")
    monkeypatch.setattr(prep.sel, "load_tables", lambda *a, **k: pytest.fail("inputs read"))
    with pytest.raises(FileExistsError):
        prep.main(argv)


def test_stage_wrapper_refuses_other_monthly_files(tmp_path):
    prep = _load_script("prepare_geo200")
    argv = _synthetic_project(tmp_path)
    argv[argv.index("--array") + 1] = str(tmp_path / "year_2023" / "2023_p02.npy")
    with pytest.raises(SystemExit, match="2023_p01"):
        prep.main(argv)


def test_stage_wrapper_refuses_changed_inputs(tmp_path):
    prep = _load_script("prepare_geo200")
    argv = _synthetic_project(tmp_path)
    with open(tmp_path / "data" / "audit.csv", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="approved hashes"):
        prep.main(argv)
    assert not (tmp_path / "data" / "geo200").exists()


def test_verifier_checks_pass_on_synthetic_checkpoint(tmp_path):
    prep, verify = _load_script("prepare_geo200"), _load_script("verify_geo200")
    argv = _synthetic_project(tmp_path)
    assert prep.main(argv) == 0
    out = tmp_path / "data" / "geo200"
    cfg = sel.load_config(tmp_path / "sel.json")
    r = verify.va.Results()
    verify.check_selection(r, np.load(tmp_path / "node_order.npy"), cfg, out, tmp_path)
    verify.check_checkpoint(r, out)
    verify.near_duplicates(r, out)
    assert not r.failed, r.failed
    assert sum(x["status"] == "PASS" for x in r.rows) >= 12
