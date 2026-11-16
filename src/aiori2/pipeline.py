"""AIORI 2.0 three-year preprocessing steps (see aiori2/README.md).

protect -> extract -> assemble -> quality -> experiments -> manifest -> verify. Each step writes atomically into the
versioned output directory and is skipped when its recorded inputs are unchanged. Existing AIORI artifacts
(nine_month data, Phase 9 and accuracy-optimisation results) are only ever read.
"""

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aiori import preprocessing, windows
from aiori2 import core

log = logging.getLogger("aiori2")


def _peak_mb():
    """Peak traced Python/NumPy heap since the CLI started (tracemalloc; memory-mapped pages are not counted)."""
    import tracemalloc
    return round(tracemalloc.get_traced_memory()[1] / 2**20) if tracemalloc.is_tracing() else None


def _out(cfg, *parts):
    return cfg["_out"].joinpath(*parts)


# ---- sensors and clients (existing assignments, never regenerated) ----

def sensors(cfg):
    from aiori.federated import iid_partition                  # the study's own IID logic and seed
    sel = pd.read_csv(cfg["_base"] / cfg["selection"])
    node = np.load(cfg["_base"] / cfg["node_order"], allow_pickle=False)
    if len(sel) != 200 or sel["station_id"].nunique() != 200:
        raise SystemExit("STOP: the selection does not hold 200 unique sensors")
    if not (node[sel["array_index"].to_numpy()] == sel["station_id"].to_numpy()).all():
        raise SystemExit("STOP: node_order[array_index] != station_id for the selection")
    iid = iid_partition(SimpleNamespace(n_sensors=len(sel)), cfg["iid_partition_seed"], cfg["n_clients"])
    iid_of = np.empty(len(sel), object)
    for name, pos in iid.items():
        iid_of[pos] = name
    out = pd.DataFrame(dict(position=np.arange(len(sel)), array_index=sel["array_index"], station_id=sel["station_id"],
                            client_geo=sel["proposed_client"], client_iid=iid_of, type_group=sel["type_group"],
                            flow_tertile=sel["flow_tertile"], district=sel["District"]))
    if out.groupby("client_geo").size().tolist() != [50] * 4 or out.groupby("client_iid").size().tolist() != [50] * 4:
        raise SystemExit("STOP: clients are not 4 x 50")
    return out


def groups(sens, column):
    return {c: np.flatnonzero(sens[column].to_numpy() == c) for c in sorted(sens[column].unique())}


# ---- protect: hash existing artifacts before anything is written ----

def protect(cfg):
    path = _out(cfg, "manifests", "protected_baseline.json")
    if path.is_file():
        log.info("protected baseline already recorded (%s); not overwritten", path)
        return core.load_json(path)
    files = {}
    for rel in cfg["protected_paths"]:
        p = cfg["_base"] / rel
        for f in sorted(p.rglob("*") if p.is_dir() else [p]):
            if f.is_file() and "__pycache__" not in f.parts:
                files[f.relative_to(cfg["_base"]).as_posix()] = dict(size=f.stat().st_size, sha256=core.sha256_file(f))
    res = dict(recorded=time.strftime("%Y-%m-%dT%H:%M:%S"), files=files)
    core.save_json(path, res)
    log.info("protected baseline: %d files", len(files))
    return res


# ---- extract: 200 selected rows, one monthly file at a time ----

def extract(cfg):
    t0 = time.time()
    sens = sensors(cfg)
    rows = sens["array_index"].to_numpy()
    order = np.argsort(rows)
    inv = np.argsort(order)
    sel_sha = core.sha256_file(cfg["_base"] / cfg["selection"])
    n_rows = len(np.load(cfg["_base"] / cfg["node_order"], allow_pickle=False))
    mpath = _out(cfg, "manifests", "extraction.json")
    man = core.load_json(mpath, {})
    for year in cfg["years"]:
        files = sorted((cfg["_base"] / cfg["raw_dirs"][str(year)]).glob(f"{year}_p[0-9][0-9].npy"))
        if [f.name for f in files] != [f"{year}_p{m:02d}.npy" for m in range(1, 13)]:
            raise SystemExit(f"STOP: {year} does not have exactly the 12 monthly files")
        for m, f in enumerate(files, 1):
            out = _out(cfg, "extracted", str(year), f.name)
            st = f.stat()
            prev = man.get(f.name)
            if (prev and prev["source"]["size"] == st.st_size and prev["source"]["mtime_ns"] == st.st_mtime_ns
                    and prev["selection_sha256"] == sel_sha and out.is_file() and core.sha256_file(out) == prev["output"]["sha256"]):
                continue
            a = np.load(f, mmap_mode="r", allow_pickle=False)
            days = (np.datetime64(f"{year}-{m:02d}") + np.timedelta64(1, "M")).astype("datetime64[D]") - np.datetime64(f"{year}-{m:02d}-01")
            expect = (n_rows, int(days.astype(int)) * cfg["steps_per_day"], 3)
            if a.shape != expect or a.dtype.kind != "f":
                raise SystemExit(f"STOP: {f.name} shape/dtype {a.shape} {a.dtype} != expected {expect} float")
            x = np.asarray(a[rows[order]])[inv]                       # sorted read (fast), then the selection order
            x32 = x.astype(np.float32)
            fc = cfg["channels"]["flow"]
            max_err = [0.0] * x.shape[2]
            if x.dtype != np.float32:
                # Flow (the target) must survive exactly; occupancy/speed decimals may only move by float32 rounding,
                # which is how the 2023 files already store them.
                if not np.array_equal(x32[..., fc].astype(x.dtype), x[..., fc], equal_nan=True):
                    raise SystemExit(f"STOP: {f.name} flow values change when converted to float32")
                if not np.allclose(x32.astype(x.dtype), x, rtol=1.2e-7, atol=0, equal_nan=True):
                    raise SystemExit(f"STOP: {f.name} values move by more than float32 rounding")
                with np.errstate(invalid="ignore"):
                    max_err = [float(np.nanmax(np.abs(x32[..., c].astype(x.dtype) - x[..., c]), initial=0)) for c in range(x.shape[2])]
            core.save_npy(out, x32)
            man[f.name] = dict(year=year, month=m, selection_sha256=sel_sha,
                               source=dict(path=f.relative_to(cfg["_base"]).as_posix(), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                           dtype=str(a.dtype), shape=list(a.shape),
                                           sha256=core.sha256_file(f) if cfg["sha256_sources"] else None),
                               conversion=dict(source_dtype=str(a.dtype), to="float32", flow_exact=True,
                                               max_abs_change_by_channel=max_err),
                               output=dict(path=out.relative_to(cfg["_out"]).as_posix(), shape=list(x32.shape), sha256=core.sha256_file(out)))
            core.save_json(mpath, man)                                # resumable: one entry per completed month
            log.info("extracted %s (%.0fs)", f.name, time.time() - t0)
    return dict(seconds=round(time.time() - t0, 1), files=len(man), peak_mb=_peak_mb())


# ---- assemble: continuous 2022-2024 timeline + calendar ----

def assemble(cfg):
    t0 = time.time()
    man = core.load_json(_out(cfg, "manifests", "extraction.json"), {})
    if len(man) != 12 * len(cfg["years"]):
        raise SystemExit("STOP: extraction incomplete")
    parts = [np.load(_out(cfg, man[k]["output"]["path"]), mmap_mode="r") for k in sorted(man, key=lambda k: (man[k]["year"], man[k]["month"]))]
    T = sum(p.shape[1] for p in parts)
    cal = core.calendar(cfg["years"], cfg["steps_per_day"], cfg["step_minutes"])
    if T != len(cal["timestamp"]):
        raise SystemExit(f"STOP: {T} extracted steps != {len(cal['timestamp'])} calendar steps")
    c = cfg["channels"]
    flow = np.concatenate([p[:, :, c["flow"]] for p in parts], axis=1)
    all3zero = np.concatenate([(p[:, :, 0] == 0) & (p[:, :, 1] == 0) & (p[:, :, 2] == 0) for p in parts], axis=1)
    nonfinite = np.concatenate([~np.isfinite(np.asarray(p)).all(axis=2) for p in parts], axis=1)
    core.save_npy(_out(cfg, "extracted", "flow_raw.npy"), flow)
    core.save_npy(_out(cfg, "flags", "all_three_zero.npy"), all3zero)
    core.save_npy(_out(cfg, "flags", "nonfinite_any_channel.npy"), nonfinite)
    for k, v in cal.items():
        core.save_npy(_out(cfg, "metadata", "calendar", f"{k}.npy"), v)
    sens = sensors(cfg)
    core.save_csv(_out(cfg, "metadata", "sensors.csv"), sens)
    core.save_json(_out(cfg, "metadata", "timeline.json"), dict(
        origin=str(cal["timestamp"][0]), end_exclusive=str(cal["timestamp"][-1] + np.timedelta64(cfg["step_minutes"], "m")),
        n_steps=int(T), steps_per_day=cfg["steps_per_day"], step_minutes=cfg["step_minutes"],
        year_start_index={str(y): core.date_index(f"{y}-01-01", f"{cfg['years'][0]}-01-01", cfg["steps_per_day"]) for y in cfg["years"]},
        timestamps=cfg["timestamps"], timezone=cfg["timezone_assumption"], channels=cfg["channels"],
        channel_definitions=cfg["channel_definitions"], sensor_order="sensor_selection.csv row order (position 0..199)"))
    return dict(seconds=round(time.time() - t0, 1), shape=list(flow.shape), peak_mb=_peak_mb())


# ---- quality: missing-data policies (masks) and quality flags, kept separate ----

def policy_missing(flow, all3zero, zero_fill_cols, min_run):
    sus = all3zero & zero_fill_cols[None, :]
    if min_run > 1:
        sus = sus & (core.run_lengths(sus) >= min_run)
    return ~np.isfinite(flow) | sus


def quality(cfg):
    t0 = time.time()
    flow = np.load(_out(cfg, "extracted", "flow_raw.npy"))
    all3zero = np.load(_out(cfg, "flags", "all_three_zero.npy"))
    year = np.load(_out(cfg, "metadata", "calendar", "year.npy"))
    zf = np.isin(year, cfg["zero_fill_years"])
    L = cfg["quality"]["long_run_steps"]
    finite = np.isfinite(flow)
    core.save_npy(_out(cfg, "flags", "nonfinite_flow.npy"), ~finite)
    zero_run = core.run_lengths(finite & (flow == 0))
    const_run = core.constant_runs(flow)
    core.save_npy(_out(cfg, "flags", "zero_run_24h.npy"), zero_run >= L)
    core.save_npy(_out(cfg, "flags", "constant_run_24h.npy"), const_run >= L)
    sens = sensors(cfg)
    rows, summary = [], {}
    years = cfg["years"]
    for name, pol in cfg["policies"].items():
        missing = policy_missing(flow, all3zero, zf, pol["all_zero_min_run_steps"])
        miss_run = core.run_lengths(missing)
        core.save_npy(_out(cfg, "masks", name, "suspected_missing.npy"), missing)
        core.save_npy(_out(cfg, "masks", name, "valid_target.npy"), ~missing)
        core.save_npy(_out(cfg, "flags", f"suspected_outage_24h_{name}.npy"), miss_run >= L)
        for y in years:
            cols = year == y
            m = missing[:, cols]
            summary.setdefault(name, {})[str(y)] = dict(
                steps=int(cols.sum()), missing_rate=round(float(m.mean()), 6),
                nonfinite_rate=round(float((~finite[:, cols]).mean()), 6),
                all_three_zero_rate=round(float(all3zero[:, cols].mean()), 6),
                sensors_with_outage_ge_24h=int((miss_run[:, cols] >= L).any(1).sum()),
                sensors_with_zero_run_ge_24h=int((zero_run[:, cols] >= L).any(1).sum()),
                sensors_with_constant_run_ge_24h=int((const_run[:, cols] >= L).any(1).sum()),
                sensors_coverage_lt_80pct=int(((~m).mean(1) < .8).sum()))
            if name == cfg["default_policy"] or name == "outage":
                for i in range(len(sens)):
                    rows.append(dict(policy=name, year=y, position=i, station_id=int(sens.station_id[i]), client_geo=sens.client_geo[i],
                                     total_steps=int(cols.sum()), valid_targets=int((~m[i]).sum()), suspected_missing=int(m[i].sum()),
                                     coverage_pct=round(100 * float((~m[i]).mean()), 3),
                                     longest_suspected_outage_steps=int(miss_run[i, cols].max()),
                                     longest_constant_run_steps=int(const_run[i, cols].max()),
                                     longest_zero_run_steps=int(zero_run[i, cols].max())))
    core.save_csv(_out(cfg, "reports", "quality_by_sensor_year.csv"), pd.DataFrame(rows))
    core.save_json(_out(cfg, "reports", "quality_summary.json"), dict(
        long_run_steps=L, note="runs are measured on the continuous 2022-2024 timeline, so a run crossing a month or year boundary counts in full",
        policies={k: v["description"] for k, v in cfg["policies"].items()}, by_policy_year=summary))
    return dict(seconds=round(time.time() - t0, 1), summary=summary, peak_mb=_peak_mb())


# ---- experiments A/B: splits, train-only statistics, causal inputs, scalers, window indices ----

def experiment_splits(cfg, exp):
    origin, spd = f"{cfg['years'][0]}-01-01", cfg["steps_per_day"]
    off = core.date_index(cfg["experiments"][exp]["train"][0], origin, spd)
    bounds = dict(train=cfg["experiments"][exp]["train"], val=cfg["val"], test=cfg["test"])
    idx = {k: (core.date_index(a, origin, spd) - off, core.date_index(b, origin, spd) - off) for k, (a, b) in bounds.items()}
    if not (idx["train"][1] == idx["val"][0] and idx["val"][1] == idx["test"][0]):
        raise SystemExit("STOP: splits are not contiguous")
    return off, idx, bounds


def fit_scalers(x, splits, sens):
    out = dict(per_sensor=preprocessing.fit_scaler(x, splits, "per_sensor"), pooled=preprocessing.fit_scaler(x, splits, "global"))
    for scope, column in (("client_geo", "client_geo"), ("client_iid", "client_iid")):
        mean, std, n_obs = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x), np.int64)
        for c, pos in groups(sens, column).items():
            s = preprocessing.fit_scaler(x[pos], splits, "global")     # each client sees only its own sensors
            mean[pos], std[pos], n_obs[pos] = s["mean"], s["std"], s["n_obs"]
        out[scope] = dict(mode=scope, fit_split="train", fit_range=list(splits["train"]), mean=mean.tolist(), std=std.tolist(),
                          n_obs=n_obs.tolist(), groups={c: p.tolist() for c, p in groups(sens, column).items()})
    return out


def window_counts(valid, splits, split, sens, cfg):
    H, Lin = cfg["horizon"], cfg["input_len"]
    allw = windows.window_index(valid, splits, split, Lin, H, 0)
    kept = windows.window_index(valid, splits, split, Lin, H, cfg["min_valid_targets"])
    full = windows.window_index(valid, splits, split, Lin, H, H)
    per = {k: np.bincount(w[:, 0], minlength=len(sens)) for k, w in (("candidates", allw), ("kept", kept), ("all_12_valid", full))}
    res = {}
    for column in ("client_geo", "client_iid"):
        for c, pos in groups(sens, column).items():
            res[f"{column}:{c}"] = {k: int(v[pos].sum()) for k, v in per.items()}
    res["total"] = dict(candidates=len(allw), kept=len(kept), all_12_valid=len(full),
                        removed_by_min_valid_targets=len(allw) - len(kept))
    return kept, res


def experiments(cfg):
    t0 = time.time()
    flow = np.load(_out(cfg, "extracted", "flow_raw.npy"), mmap_mode="r")
    ts = np.load(_out(cfg, "metadata", "calendar", "timestamp.npy"))
    sens = sensors(cfg)
    report = {}
    for pol in cfg["policies"]:
        valid_all = np.load(_out(cfg, "masks", pol, "valid_target.npy"), mmap_mode="r")
        for exp in cfg["experiments"]:
            d = _out(cfg, "experiments", pol, exp)
            off, splits, bounds = experiment_splits(cfg, exp)
            valid = np.asarray(valid_all[:, off:])
            x = np.where(valid, np.asarray(flow[:, off:], dtype=np.float64), np.nan)      # policy-missing -> NaN
            tod = preprocessing.fit_tod_median(x, splits, cfg["steps_per_day"])         # training observations only
            filled, imputed = preprocessing.causal_fill(x, splits, tod, cfg["max_ffill_steps"], cfg["steps_per_day"])
            scalers = fit_scalers(x, splits, sens)
            core.save_npy(d / "filled_inputs.npy", filled.astype(np.float32))
            core.save_npy(d / "input_imputed.npy", imputed)
            core.save_npy(d / "tod_median_train.npy", tod)
            counts = {}
            for split in ("train", "val", "test"):
                kept, counts[split] = window_counts(valid, splits, split, sens, cfg)
                s0, s1 = splits[split]
                assert ((kept[:, 1] >= s0) & (kept[:, 1] + cfg["horizon"] <= s1) & (kept[:, 1] >= cfg["input_len"])).all()
                core.save_npy(d / "windows" / f"{split}.npy", kept.astype(np.int32))
            meta = dict(experiment=exp, policy=pol, description=cfg["experiments"][exp]["description"],
                        timeline_offset=off, first_timestamp=str(ts[off]), splits_index=splits, splits_dates=bounds,
                        input_len=cfg["input_len"], horizon=cfg["horizon"], min_valid_targets=cfg["min_valid_targets"],
                        max_ffill_steps=cfg["max_ffill_steps"], input_channels=cfg["input_channels"],
                        window_rule="t0 = first target step; inputs [t0-12, t0) may precede the split but never the experiment start; targets [t0, t0+12) lie inside the split",
                        config_sha256=cfg["_config_sha256"], window_counts=counts)
            core.save_json(d / "experiment.json", meta)
            core.save_json(d / "scalers.json", dict(
                fit_on="training split only, policy-valid observations only", train_dates=bounds["train"], fit_range=list(splits["train"]),
                config_sha256=cfg["_config_sha256"], arm_default_scope=cfg["arm_scaler_scope"],
                inverse_transform="raw_flow = z * std[sensor] + mean[sensor]; metrics are computed on raw flow with the policy valid-target mask",
                scopes=scalers))
            report[f"{pol}/{exp}"] = {k: v["total"] for k, v in counts.items()}
            log.info("experiment %s/%s done (%.0fs)", pol, exp, time.time() - t0)
            del x, filled, imputed, valid
    return dict(seconds=round(time.time() - t0, 1), windows=report, peak_mb=_peak_mb())


# ---- manifest + verify ----

CODE = ["aiori2/__init__.py", "aiori2/__main__.py", "aiori2/core.py", "aiori2/pipeline.py", "aiori2/data.py", "aiori2/README.md",
        "aiori2/configs/preprocess_v1.json", "aiori/__init__.py", "aiori/config.py", "aiori/preprocessing.py", "aiori/windows.py",
        "aiori/federated.py", "aiori/dataset.py", "aiori/evaluation.py", "aiori/metrics.py", "aiori/training.py", "aiori/model.py"]


def manifest(cfg):
    pol = cfg["default_policy"]
    shared = ["metadata/sensors.csv", "metadata/timeline.json", "extracted/flow_raw.npy", f"masks/{pol}/valid_target.npy",
              "manifests/extraction.json", "reports/quality_summary.json"] + \
             [p.relative_to(cfg["_out"]).as_posix() for p in sorted(_out(cfg, "metadata", "calendar").glob("*.npy"))]
    per_exp = [p.relative_to(cfg["_out"]).as_posix() for e in cfg["experiments"] for p in sorted(_out(cfg, "experiments", pol, e).rglob("*")) if p.is_file()]
    files = {}
    for rel in shared + per_exp:
        p = _out(cfg, rel)
        files[rel] = dict(size=p.stat().st_size, sha256=core.sha256_file(p))
    code = {rel: core.sha256_file(cfg["_base"] / rel) for rel in CODE if (cfg["_base"] / rel).is_file()}
    m = dict(purpose="minimum files to run GPU training on the DGX (default policy only; nothing is transferred in Phase 2)",
             policy=pol, data_root=cfg["out_dir"], data_files=files, data_bytes=sum(v["size"] for v in files.values()),
             code_root="src", code_files=code, extra=["requirements.txt (+ CUDA torch build matching the DGX driver)"],
             not_needed=["raw year_2022/2023/2024 (~102 GB)", "extracted/<year>/ monthly 3-channel reference arrays",
                         "the outage-policy variant (masks/outage, experiments/outage)"],
             verify="python -m aiori2 verify --base-dir <src> after copying; it re-hashes every listed file")
    core.save_json(_out(cfg, "manifests", "dgx_transfer_manifest.json"), m)
    return dict(data_files=len(files), data_gb=round(m["data_bytes"] / 1e9, 3))


def verify(cfg):
    problems = []
    base = core.load_json(_out(cfg, "manifests", "protected_baseline.json"), {}).get("files", {})
    for rel, m in base.items():
        p = cfg["_base"] / rel
        if not p.is_file() or core.sha256_file(p) != m["sha256"]:
            problems.append(f"protected file changed or missing: {rel}")
    for k, m in core.load_json(_out(cfg, "manifests", "extraction.json"), {}).items():
        p = _out(cfg, m["output"]["path"])
        if not p.is_file() or core.sha256_file(p) != m["output"]["sha256"]:
            problems.append(f"extracted file damaged: {k}")
    dgx = core.load_json(_out(cfg, "manifests", "dgx_transfer_manifest.json"), {})
    for rel, m in dgx.get("data_files", {}).items():
        p = _out(cfg, rel)
        if not p.is_file() or core.sha256_file(p) != m["sha256"]:
            problems.append(f"transfer file damaged: {rel}")
        elif p.suffix == ".npy":
            np.load(p, mmap_mode="r", allow_pickle=False)
    return dict(protected_files=len(base), problems=problems, ok=not problems)


STEPS = dict(protect=protect, extract=extract, assemble=assemble, quality=quality, experiments=experiments, manifest=manifest, verify=verify)
