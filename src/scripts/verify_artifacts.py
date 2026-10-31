"""Rerunnable, read-only verification of raw inputs and generated artifacts on the real data.

Run from the project root:
    .venv/Scripts/python.exe scripts/verify_artifacts.py                    # all checks
    .venv/Scripts/python.exe scripts/verify_artifacts.py --record-baseline  # once: write raw-hash baseline

Exit code 0 when every check passes, 1 otherwise. Results are also written to
data/processed/verification_results.json.

The raw-file hash baseline (data/raw_integrity_baseline.json) was recorded locally on
2026-10-24, after the files were downloaded. It is NOT a hash of the original download
or a publisher checksum; it only detects changes made after it was recorded.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiori import config, data_io, flow_audit, preprocessing, splits, windows  # noqa: E402

BASELINE_PATH = ROOT / "data" / "raw_integrity_baseline.json"
RAW_FILES = [config.ARRAY_PATH, config.NODE_ORDER_PATH, config.METADATA_PATH]
P = config.PROCESSED_DIR
PREPROCESSED_RUNS = {
    "primary": (P, P / "sensor_selection.csv"),
    "missing_validation": (P / "missing_validation", P / "missing_validation_selection.csv"),
}
HANDOFF_CLIENT_COUNTS = {"client_0_la_ventura": 4857, "client_1_bay_area": 4072,
                         "client_2_coastal_south_central": 2974, "client_3_inland_central": 5069}
HANDOFF_CLIENT_DISTRICTS = {"client_0_la_ventura": [7], "client_1_bay_area": [4],
                            "client_2_coastal_south_central": [5, 12], "client_3_inland_central": [3, 6, 8, 10]}


class Results:
    def __init__(self):
        self.rows = []

    def check(self, section, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        self.rows.append(dict(section=section, check=name, status=status, detail=str(detail)))
        print(f"[{status}] {section}: {name}" + (f" - {detail}" if detail else ""))
        return ok

    def info(self, section, name, detail):
        self.rows.append(dict(section=section, check=name, status="INFO", detail=str(detail)))
        print(f"[INFO] {section}: {name} - {detail}")

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def record_baseline():
    if BASELINE_PATH.exists():
        raise SystemExit(f"Baseline already exists at {BASELINE_PATH}; refusing to overwrite.")
    files = {}
    for p in RAW_FILES:
        st = p.stat()
        files[str(p.relative_to(ROOT)).replace("\\", "/")] = dict(
            sha256=sha256_file(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
    BASELINE_PATH.write_text(json.dumps(dict(
        recorded_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        provenance="Local baseline recorded during the project, after download. NOT the hash of the "
                   "original download and NOT a publisher checksum; detects changes after recording only.",
        files=files), indent=2), encoding="utf-8")
    print(f"Wrote {BASELINE_PATH}")


def check_raw(r):
    s = "raw integrity"
    if not BASELINE_PATH.exists():
        r.check(s, "baseline file present", False, f"{BASELINE_PATH} missing; run with --record-baseline")
        return
    base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    r.info(s, "baseline provenance", f"recorded {base['recorded_at_utc']}; local baseline, not original-download hash")
    for rel, exp in base["files"].items():
        p = ROOT / rel
        if not r.check(s, f"{rel} exists", p.is_file()):
            continue
        st = p.stat()
        r.check(s, f"{rel} size", st.st_size == exp["size"], st.st_size)
        r.check(s, f"{rel} sha256 matches baseline", sha256_file(p) == exp["sha256"], exp["sha256"][:16] + "...")
        if st.st_mtime_ns != exp["mtime_ns"]:
            r.info(s, f"{rel} mtime", "differs from baseline (content hash is authoritative)")


def check_inputs(r):
    s = "inputs"
    arr = data_io.load_traffic_array(config.ARRAY_PATH)
    r.check(s, "2023_p01 shape", tuple(arr.shape) == config.EXPECTED_P01_SHAPE, arr.shape)
    r.check(s, "2023_p01 dtype float32", arr.dtype == np.float32, arr.dtype)
    node_order = data_io.load_node_order(config.NODE_ORDER_PATH, n_sensors=arr.shape[0])
    r.check(s, "node_order unique, length 16,972", len(node_order) == 16972)
    aligned = data_io.align_metadata(data_io.load_metadata(config.METADATA_PATH), node_order)
    r.check(s, "metadata aligns to node_order", np.array_equal(aligned["station_id"].to_numpy(), node_order))
    r.info(s, "row-to-ID mapping", "supported but unconfirmed: array row i <-> node_order[i] is not proven")
    return arr, node_order


def check_assignment(r, node_order):
    s = "client assignment"
    path = P / "proposed_client_assignment.csv"
    if not r.check(s, "file present", path.is_file(), path):
        return None
    asg = pd.read_csv(path)
    r.check(s, "order == node_order", np.array_equal(asg["station_id"].to_numpy(), node_order))
    r.check(s, "client counts match handoff",
            asg["proposed_client"].value_counts().to_dict() == HANDOFF_CLIENT_COUNTS,
            asg["proposed_client"].value_counts().sort_index().tolist())
    districts = asg.groupby("proposed_client")["District"].agg(lambda x: sorted(x.unique().tolist())).to_dict()
    r.check(s, "districts per client match handoff", districts == HANDOFF_CLIENT_DISTRICTS, districts)
    return asg


def check_audit(r, arr, split):
    s = "audit"
    audit = pd.read_csv(P / "flow_sensor_audit.csv", keep_default_na=False, na_values=[""])
    audit["exclusion_reason"] = audit["exclusion_reason"].fillna("")
    r.check(s, "one row per sensor", len(audit) == arr.shape[0] and audit["array_index"].is_unique, len(audit))
    a, b = split["train"]
    r.check(s, "eligibility statistics period == train split",
            (audit["stats_period"] == f"train steps [{a}, {b})").all(), f"[{a}, {b})")
    m = lambda c: audit[f"month_{c}"]  # noqa: E731
    month = {
        "nan_cells": int(m("missing_count").sum() - m("inf_count").sum()), "zero_cells": int(m("zero_count").sum()),
        "fully_observed": int((m("missing_count") == 0).sum()),
        "partially_observed": int(((m("missing_count") > 0) & ~m("fully_missing")).sum()),
        "fully_missing": int(m("fully_missing").sum()), "constant": int(m("constant").sum()),
    }
    r.check(s, "whole-month Ch0 counts match handoff", month == flow_audit.HANDOFF_P01_FLOW, month)
    recomputed = flow_audit.assign_eligibility(audit.drop(columns=["eligible", "exclusion_reason"]), 99.0, 50.0)
    r.check(s, "stored eligibility == recomputed from train columns (99% / 50%)",
            (recomputed["eligible"].to_numpy() == audit["eligible"].to_numpy()).all(), int(audit["eligible"].sum()))
    r.info(s, "train-period exact duplicates",
           f"{int(audit['duplicate_series'].sum())} sensors in {audit.loc[audit.duplicate_series, 'series_sha256'].nunique()} groups "
           f"(whole month: {int(m('duplicate_series').sum())})")
    r.info(s, "exclusion reasons", audit["exclusion_reason"].replace("", "eligible").value_counts().to_dict())
    return audit


def check_manifests(r, arr, node_order, asg, audit, split):
    s = "manifests"
    a, b = split["train"]
    prim = pd.read_csv(P / "sensor_selection.csv")
    r.check(s, "primary: 40 sensors", len(prim) == 40, len(prim))
    r.check(s, "primary: 10 per client", prim.groupby("proposed_client").size().tolist() == [10] * 4,
            prim.groupby("proposed_client").size().to_dict())
    r.check(s, "primary: unique array_index / station_id / series hash",
            prim["array_index"].is_unique and prim["station_id"].is_unique and prim["series_sha256"].is_unique)
    r.check(s, "primary: all eligible", prim["eligible"].all())
    r.check(s, "primary: station_id == node_order[array_index]",
            np.array_equal(node_order[prim["array_index"]], prim["station_id"].to_numpy()))
    if asg is not None:
        r.check(s, "primary: client agrees with assignment file",
                (asg.loc[prim["array_index"], "proposed_client"].to_numpy() == prim["proposed_client"].to_numpy()).all())
    by = prim.groupby("proposed_client")["District"].agg(lambda x: sorted(set(x))).to_dict()
    r.check(s, "primary: every District of every client represented", by == HANDOFF_CLIENT_DISTRICTS, by)
    coastal = prim[prim.proposed_client == "client_2_coastal_south_central"]["District"]
    r.check(s, "primary: coastal has >=1 District 5 and >=1 District 12",
            (coastal == 5).sum() >= 1 and (coastal == 12).sum() >= 1,
            {5: int((coastal == 5).sum()), 12: int((coastal == 12).sum())})
    r.info(s, "primary: client/District counts", prim.groupby(["proposed_client", "District"]).size().to_dict())
    rows = prim["array_index"].to_numpy()
    flow = np.asarray(arr[rows][:, :, config.FLOW])
    r.check(s, "primary: training coverage recomputed from raw is >= 99%",
            (np.isfinite(flow[:, a:b]).mean(axis=1) >= 0.99).all())
    r.info(s, "primary: whole-January missing cells (outcome, not a selection rule)", int((~np.isfinite(flow)).sum()))

    vpath = P / "missing_validation_selection.csv"
    if not r.check(s, "validation manifest present", vpath.is_file(), vpath):
        return
    val = pd.read_csv(vpath)
    r.check(s, "validation: disjoint from primary", set(val["array_index"]).isdisjoint(rows))
    vflow = np.asarray(arr[val["array_index"].to_numpy()][:, :, config.FLOW])
    cov = 100 * np.isfinite(vflow).mean(axis=1)
    r.check(s, "validation: January coverage in [99, 100)", ((cov >= 99) & (cov < 100)).all(),
            [round(c, 3) for c in cov])
    eval_missing = (~np.isfinite(vflow[:, b:])).sum(axis=1)
    r.check(s, "validation: every sensor has >=1 missing step in val/test (raw)", (eval_missing >= 1).all(),
            eval_missing.tolist())
    r.check(s, "validation: station_id == node_order[array_index]",
            np.array_equal(node_order[val["array_index"]], val["station_id"].to_numpy()))


def check_preprocessed(r, arr, label, out_dir, manifest):
    s = f"preprocessed[{label}]"
    files = ["split_config.json", "scaler_parameters.json", "pilot_preprocessed.npz", "window_index.npz"]
    missing = [f for f in files if not (out_dir / f).is_file()]
    if not r.check(s, "artifacts present", not missing, missing or out_dir):
        return
    sel = pd.read_csv(manifest)
    split = splits.load_split_config(out_dir / "split_config.json")
    r.check(s, "split == make_splits(8928)", split == splits.make_splits(arr.shape[1]), split)
    sc = json.loads((out_dir / "scaler_parameters.json").read_text(encoding="utf-8"))
    d = np.load(out_dir / "pilot_preprocessed.npz")
    w = np.load(out_dir / "window_index.npz")
    rows = sel["array_index"].to_numpy()
    r.check(s, "sensor order == manifest", np.array_equal(d["array_index"], rows) and sc["array_index"] == rows.tolist())
    raw = np.asarray(arr[rows][:, :, config.FLOW], dtype=np.float64)
    r.check(s, "flow_raw == memory-mapped source", np.array_equal(d["flow_raw"], raw.astype(np.float32), equal_nan=True))

    a, b = split["train"]
    train = raw[:, a:b]
    r.info(s, "scaler mode", sc["mode"])
    if sc["mode"] == "per_sensor":
        mean, std = np.nanmean(train, axis=1), np.nanstd(train, axis=1)
    else:
        mean, std = np.full(len(raw), np.nanmean(train)), np.full(len(raw), np.nanstd(train))
    std = np.where(std < preprocessing.MIN_STD, 1.0, std)
    r.check(s, "scaler mean/std == train-only recompute", np.allclose(sc["mean"], mean) and np.allclose(sc["std"], std))
    r.check(s, "scaler fit_range == train split", sc["fit_range"] == [a, b])
    r.check(s, "scaler n_obs == observed train cells", sc["n_obs"] == np.isfinite(train).sum(axis=1).tolist())
    r.info(s, "scaler differs from full-month mean (sensors)", int((~np.isclose(sc["mean"], np.nanmean(raw, axis=1))).sum()))

    obs = np.isfinite(raw)
    r.check(s, "observed_mask == isfinite(raw)", np.array_equal(d["observed_mask"], obs))
    r.check(s, "targets NaN exactly where raw missing", np.array_equal(np.isnan(d["targets_scaled"]), ~obs))
    back = preprocessing.inverse_transform(d["targets_scaled"], sc)
    r.check(s, "inverse_transform(targets) == raw (observed cells, atol 1e-3)",
            np.allclose(back[obs], raw[obs], atol=1e-3), f"max abs err {np.nanmax(np.abs(back - raw)):.2e}")
    r.check(s, "input_imputed == ~observed", np.array_equal(d["input_imputed"], ~obs))
    r.check(s, "inputs finite everywhere", np.isfinite(d["inputs_scaled"]).all())
    r.check(s, "inputs == targets where observed (no change to real values)",
            np.allclose(d["inputs_scaled"][obs], d["targets_scaled"][obs], atol=1e-5))
    r.check(s, "zero readings flagged observed", obs[raw == 0].all(), int((raw == 0).sum()))
    tod = preprocessing.fit_tod_median(raw, split)
    r.check(s, "tod_median == train-only recompute", np.allclose(d["tod_median"], tod, equal_nan=True))
    filled, _ = preprocessing.causal_fill(raw, split, tod)
    r.check(s, "inputs == causal_fill recompute", np.allclose(d["inputs_scaled"], preprocessing.transform(filled, sc), atol=1e-4))

    for name in splits.SPLIT_NAMES:
        start, end = split[name]
        idx = w[name]
        t0 = idx[:, 1]
        r.check(s, f"{name}: windows == recomputed index", np.array_equal(idx, windows.window_index(obs, split, name)), len(idx))
        r.check(s, f"{name}: all targets inside split, inputs start >= 0",
                bool(((t0 >= start) & (t0 + windows.HORIZON <= end) & (t0 - windows.INPUT_LEN >= 0)).all()))
        per_sensor = (end - max(start, windows.INPUT_LEN)) - windows.HORIZON + 1
        if obs.all():
            r.check(s, f"{name}: count == closed form ({per_sensor}/sensor x {len(rows)})", len(idx) == per_sensor * len(rows))
        _, _, M = windows.gather_windows(d["inputs_scaled"], d["input_imputed"], d["targets_scaled"], obs, idx)
        r.info(s, f"{name}: missing cells / imputed inputs / masked target cells / windows with a masked target",
               f"{int((~obs[:, start:end]).sum())} / {int(d['input_imputed'][:, start:end].sum())} / "
               f"{int((~M).sum())} / {int((~M).any(axis=1).sum())}")
    if label == "missing_validation":
        val_test = ~obs[:, split["val"][0]:]
        r.check(s, "real missing values exist in val/test", val_test.any(), int(val_test.sum()))
        _, _, Mt = windows.gather_windows(d["inputs_scaled"], d["input_imputed"], d["targets_scaled"], obs, w["test"])
        _, _, Mv = windows.gather_windows(d["inputs_scaled"], d["input_imputed"], d["targets_scaled"], obs, w["val"])
        r.check(s, "target masking exercised on real data (val+test)", int((~Mt).sum() + (~Mv).sum()) > 0,
                int((~Mt).sum() + (~Mv).sum()))
        r.check(s, "input fill exercised on real data", bool(d["input_imputed"].any()), int(d["input_imputed"].sum()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--record-baseline", action="store_true", help="write the raw-file hash baseline (once)")
    args = p.parse_args(argv)
    if args.record_baseline:
        record_baseline()
        return 0

    r = Results()
    check_raw(r)
    arr, node_order = check_inputs(r)
    asg = check_assignment(r, node_order)
    split = splits.make_splits(arr.shape[1])
    audit = check_audit(r, arr, split)
    check_manifests(r, arr, node_order, asg, audit, split)
    for label, (out_dir, manifest) in PREPROCESSED_RUNS.items():
        check_preprocessed(r, arr, label, out_dir, manifest)

    n_pass = sum(x["status"] == "PASS" for x in r.rows)
    out = P / "verification_results.json"
    out.write_text(json.dumps(dict(run_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                   passed=n_pass, failed=len(r.failed), results=r.rows), indent=2), encoding="utf-8")
    print(f"\n{n_pass} passed, {len(r.failed)} failed. Results: {out}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
