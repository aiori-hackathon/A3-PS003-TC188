"""AIORI 2.0 Phase 3: validate the four arms x two training histories on the three-year data (no model training).

    python -m aiori2.validate                 # all checks, CPU smoke test; writes the Phase 3 report and manifests
    python -m aiori2.validate --smoke-only    # just the tensor smoke test (run with the CUDA environment for the GPU check)

Writes only new files: <out>/phase3/ (sensitivity mask, report) and aiori2/configs/experiments/*.json (8 manifests).
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from aiori import preprocessing
from aiori2 import arms, core, pipeline
from aiori2.data import ThreeYearWindows, raw_metrics

POLICY = "conservative"
SEED = 20261117


def check(cond, what, problems):
    if not cond:
        problems.append(what)
    return bool(cond)


# ---- B: sensor identity and partitions ----

def sensors_and_partitions(cfg, P):
    from aiori.federated import iid_partition
    from types import SimpleNamespace
    root = cfg["_out"]
    sel = pd.read_csv(cfg["_base"] / cfg["selection"])
    node = np.load(cfg["_base"] / cfg["node_order"], allow_pickle=False)
    sens = pd.read_csv(root / "metadata" / "sensors.csv")
    flow = np.load(root / "extracted" / "flow_raw.npy", mmap_mode="r")
    m1 = np.load(root / "extracted" / "2023" / "2023_p01.npy", mmap_mode="r")
    raw = np.load(cfg["_base"] / cfg["raw_dirs"]["2023"] / "2023_p01.npy", mmap_mode="r")
    r = {}
    r["n_ids"] = check(len(sens) == 200, "not 200 sensors", P) and len(sens)
    r["unique_ids"] = check(sens.station_id.nunique() == 200, "duplicate station ids", P)
    r["station_to_array_index_preserved"] = check((node[sens.array_index] == sens.station_id).all() and (sens.array_index == sel.array_index).all()
                                                  and (sens.station_id == sel.station_id).all(), "station/array_index mapping changed", P)
    probe = np.array([0, 1000, 5000, 8927])
    r["row_order_matches_raw"] = check(np.array_equal(np.asarray(m1[:, probe, 0]), np.asarray(raw[np.sort(sel.array_index)][np.argsort(np.argsort(sel.array_index))][:, probe, 0]), equal_nan=True)
                                       and np.array_equal(np.asarray(flow[:, 105120 + probe]), np.asarray(m1[:, probe, 0]), equal_nan=True),
                                       "extracted rows do not follow the selection order", P)
    parts = {}
    for col in ("client_geo", "client_iid"):
        g = sens.groupby(col)["position"].apply(list).to_dict()
        allpos = np.concatenate([np.asarray(v) for v in g.values()])
        parts[col] = dict(clients=len(g), sizes={k: len(v) for k, v in g.items()},
                          covers_all_200=bool(np.array_equal(np.sort(allpos), np.arange(200))),
                          duplicates=int(len(allpos) - len(set(allpos.tolist()))),
                          omitted=int(200 - len(set(allpos.tolist()))))
        check(parts[col]["clients"] == 4 and set(parts[col]["sizes"].values()) == {50} and parts[col]["covers_all_200"]
              and parts[col]["duplicates"] == 0, f"{col} partition not 4 x 50 disjoint cover", P)
    r["geo_equals_selection_csv"] = check((sens.client_geo == sel.proposed_client).all(), "geographic clients differ from selection", P)
    iid = iid_partition(SimpleNamespace(n_sensors=200), 20261027, 4)
    r["iid_seed_reproduces"] = check(all((sens.client_iid.to_numpy()[p] == n).all() for n, p in iid.items()), "IID seed does not reproduce", P)
    r["partitions"] = parts
    r["year_identity"] = "row alignment across years is statistically supported (Phase 1), not independently proven"
    return r


# ---- C: arms and client isolation ----

def scaler_refit(cfg, history, positions, scope):
    """Re-fit a scaler for `positions` from training data only (independent of the stored artifact)."""
    off, splits, _ = pipeline.experiment_splits(cfg, history)
    root = cfg["_out"]
    f = np.load(root / "extracted" / "flow_raw.npy", mmap_mode="r")
    v = np.load(root / "masks" / POLICY / "valid_target.npy", mmap_mode="r")
    a, b = off + splits["train"][0], off + splits["train"][1]
    x = np.where(v[positions, a:b], np.asarray(f[positions, a:b], np.float64), np.nan)
    return dict(mean=float(np.nanmean(x)), std=float(np.nanstd(x))) if scope != "per_sensor" else None


def arms_check(cfg, P):
    root = cfg["_out"]
    res = {}
    for arm, spec in arms.ARMS.items():
        st = arms.streams(root, arm)
        allpos = np.concatenate([s.sensors for s in st])
        a = dict(label=spec["label"], streams=len(st), aggregation=spec["aggregation"], scaler_scope=spec["scaler"],
                 stream_sizes={s.name: len(s.sensors) for s in st},
                 covers_all_200=bool(np.array_equal(np.sort(allpos), np.arange(200))), overlap=int(len(allpos) - len(set(allpos.tolist()))))
        check(a["covers_all_200"] and a["overlap"] == 0, f"{arm}: streams do not partition the 200 sensors", P)
        check(len(st) == (1 if arm == "centralised" else 4), f"{arm}: wrong number of streams", P)
        for h in arms.HISTORIES:
            sc = json.loads((root / "experiments" / POLICY / h / "scalers.json").read_text())["scopes"][spec["scaler"]]
            iso = []
            for s in st:
                ref = scaler_refit(cfg, h, s.sensors, spec["scaler"])
                mean, std = np.asarray(sc["mean"])[s.sensors], np.asarray(sc["std"])[s.sensors]
                iso.append(bool(np.allclose(mean, ref["mean"], rtol=1e-9) and np.allclose(std, ref["std"], rtol=1e-9)))
                ds = ThreeYearWindows(root, POLICY, h, "train", spec["scaler"], sensors=s.sensors)
                take = arms.epoch_sample(len(ds), 256, SEED, 0)
                X, Y, M, sen = ds.get(take)
                check(np.isin(sen, s.sensors).all(), f"{arm}/{h}/{s.name}: batch has foreign sensors", P)
                check(np.isin(np.asarray(ds.index[:, 0]), s.sensors).all(), f"{arm}/{h}/{s.name}: index has foreign sensors", P)
            a[f"scaler_matches_refit_on_own_training_data_{h}"] = check(all(iso), f"{arm}/{h}: stored scaler != own-data refit", P)
        res[arm] = a
    return res


# ---- D: A/B consistency and leakage ----

def ab_check(cfg, P):
    root = cfg["_out"]
    meta = {h: json.loads((root / "experiments" / POLICY / h / "experiment.json").read_text()) for h in arms.HISTORIES}
    ts = np.load(root / "metadata" / "calendar" / "timestamp.npy")
    r = dict(train_dates={h: meta[h]["splits_dates"]["train"] for h in meta},
             first_timestamp={h: meta[h]["first_timestamp"] for h in meta})
    check(meta["A"]["splits_dates"]["train"] == ["2023-01-01", "2024-01-01"], "A train dates", P)
    check(meta["B"]["splits_dates"]["train"] == ["2022-01-01", "2024-01-01"], "B train dates", P)
    for split in ("val", "test"):
        w = {h: np.load(root / "experiments" / POLICY / h / "windows" / f"{split}.npy").astype(np.int64) for h in meta}
        absw = {h: np.stack([w[h][:, 0], w[h][:, 1] + meta[h]["timeline_offset"]], 1) for h in meta}
        same = np.array_equal(absw["A"], absw["B"])
        check(same, f"{split} windows differ between A and B", P)
        r[f"{split}_windows_identical"] = same
        r[f"{split}_first_last_target_time"] = [str(ts[absw["A"][:, 1].min()]), str(ts[absw["A"][:, 1].max() + 11])]
    for h in meta:
        off = meta[h]["timeline_offset"]
        for split in ("train", "val", "test"):
            w = np.load(root / "experiments" / POLICY / h / "windows" / f"{split}.npy", mmap_mode="r")
            s0, s1 = meta[h]["splits_index"][split]
            t0min, t0max = int(w[:, 1].min()), int(w[:, 1].max())
            check(t0min >= s0 and t0max + 12 <= s1, f"{h}/{split}: target crosses a split boundary", P)
            check(t0min - 12 >= 0, f"{h}/{split}: inputs reach before the experiment start", P)
            r[f"{h}_{split}_earliest_input_time"] = str(ts[t0min - 12 + off])
    check(r["A_train_earliest_input_time"].startswith("2023-01-01"), "A reads 2022", P)
    # inputs near the validation boundary: A and B fill identically except where a training fallback was used
    fa = np.load(root / "experiments" / POLICY / "A" / "filled_inputs.npy", mmap_mode="r")
    fb = np.load(root / "experiments" / POLICY / "B" / "filled_inputs.npy", mmap_mode="r")
    ia = np.load(root / "experiments" / POLICY / "A" / "input_imputed.npy", mmap_mode="r")
    oa, ob = meta["A"]["timeline_offset"], meta["B"]["timeline_offset"]
    v0 = meta["B"]["splits_index"]["val"][0]
    xa, xb = np.asarray(fa[:, v0 + ob - oa - 12:]), np.asarray(fb[:, v0 - 12:])      # same absolute times in both
    diff = xa != xb
    r["val_test_inputs_differ_A_vs_B"] = dict(positions=int(diff.sum()), all_at_imputed_positions=bool(np.asarray(ia[:, v0 + ob - oa - 12:])[diff].all()),
                                              note="differences can only come from each history's own training fallback statistics")
    check(r["val_test_inputs_differ_A_vs_B"]["all_at_imputed_positions"], "A/B val/test inputs differ at observed positions", P)
    for h in meta:
        sc = json.loads((root / "experiments" / POLICY / h / "scalers.json").read_text())
        check(sc["fit_range"][1] == meta[h]["splits_index"]["val"][0], f"{h}: scaler fit range reaches past training", P)
    r["evaluation_mask"] = "both histories read the same masks/conservative/valid_target.npy for targets"
    return r


# ---- E/F: scalers, inverse transform, metrics, loader contract ----

def loader_check(cfg, P):
    root = cfg["_out"]
    flow = np.load(root / "extracted" / "flow_raw.npy", mmap_mode="r")
    ts = np.load(root / "metadata" / "calendar" / "timestamp.npy")
    res = {}
    for h in arms.HISTORIES:
        sc = json.loads((root / "experiments" / POLICY / h / "scalers.json").read_text())["scopes"]
        for scope, s in sc.items():
            ok = np.isfinite(s["mean"]).all() and np.isfinite(s["std"]).all() and (np.asarray(s["std"]) > 0).all()
            check(ok, f"{h}/{scope}: non-finite or zero scaler", P)
        meta = json.loads((root / "experiments" / POLICY / h / "experiment.json").read_text())
        for arm, spec in arms.ARMS.items():
            for split in ("train", "val", "test"):
                st = arms.streams(root, arm)[0]
                ds = ThreeYearWindows(root, POLICY, h, split, spec["scaler"], sensors=st.sensors)
                take = arms.epoch_sample(len(ds), 512, SEED, 1)
                X, Y, M, s = ds.get(take)
                w = np.asarray(ds.index[take]).astype(np.int64)
                cols = w[:, 1:2] + np.arange(12) + meta["timeline_offset"]
                rawY = np.asarray(flow[s[:, None], cols])
                t_first = ds.target_timestamps(take, ts)
                s0, s1 = meta["splits_index"][split]
                k = f"{h}/{arm}/{split}"
                check(X.shape == (len(take), 12, 1) and X.dtype == np.float32, f"{k}: X shape/dtype", P)
                check(Y.shape == M.shape == (len(take), 12), f"{k}: Y/M shape", P)
                check(np.isfinite(X).all(), f"{k}: non-finite inputs", P)
                check(np.array_equal(np.isfinite(Y), M), f"{k}: mask != finite targets", P)
                check(np.array_equal(Y[M], rawY[M].astype(np.float32)), f"{k}: targets are not the raw flow", P)
                check(((w[:, 1] >= s0) & (w[:, 1] + 12 <= s1)).all(), f"{k}: window outside split", P)
                z = ds.scale(Y, s)
                back = ds.inverse(z, s)
                check(np.allclose(back[M], Y[M], atol=1e-3), f"{k}: inverse transform", P)
                m = raw_metrics(Y, np.where(M, Y, 0), M)
                check(m["mae"] == 0 and m["rmse"] == 0 and m["wape_pct"] == 0, f"{k}: identity metrics not zero", P)
                if split == "train" and arm == "centralised":
                    res[k] = dict(X=list(X.shape), Y=list(Y.shape), M=list(M.shape), valid_target_share=round(float(M.mean()), 4),
                                  first_target_time_sample=str(t_first[0]))
        dso = ThreeYearWindows(root, POLICY, h, "train", "pooled", input_channels=["flow", "observed"])
        X, Y, M, s = dso.get(arms.epoch_sample(len(dso), 512, SEED, 2))
        check(X.shape[-1] == 2 and set(np.unique(X[..., 1])) <= {0.0, 1.0}, "observed-flag channel", P)
        res[f"{h}/observed_channel_imputed_share"] = round(float(1 - X[..., 1].mean()), 5)
    res["metric_zero_denominator"] = raw_metrics(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2), bool))
    return res


# ---- G: sampling fairness ----

def fairness(cfg):
    root = cfg["_out"]
    sens = pd.read_csv(root / "metadata" / "sensors.csv")
    out = {}
    for h in arms.HISTORIES:
        for split in ("train", "val", "test"):
            w = np.load(root / "experiments" / POLICY / h / "windows" / f"{split}.npy", mmap_mode="r")
            per = np.bincount(np.asarray(w[:, 0]), minlength=200)
            med = float(np.median(per))
            low = np.flatnonzero(per < 0.5 * med)
            out[f"{h}/{split}"] = dict(windows=int(len(w)), per_sensor_min=int(per.min()), per_sensor_median=med, per_sensor_max=int(per.max()),
                                       sensors_below_half_median=[int(sens.station_id[i]) for i in low],
                                       share_by_geo_client={c: round(float(per[g].sum() / per.sum()), 4) for c, g in pipeline.groups(sens, "client_geo").items()},
                                       share_by_iid_client={c: round(float(per[g].sum() / per.sum()), 4) for c, g in pipeline.groups(sens, "client_iid").items()})
    a, b = arms.epoch_sample(10_000_000, 1000, SEED, 3), arms.epoch_sample(10_000_000, 1000, SEED, 3)
    out["epoch_sample_deterministic"] = bool(np.array_equal(a, b)) and not np.array_equal(a, arms.epoch_sample(10_000_000, 1000, SEED, 4))
    return out


# ---- H: evaluation-only constant-run sensitivity mask ----

def sensitivity(cfg):
    root = cfg["_out"]
    d = root / "phase3" / "sensitivity" / "constant_run_24h"
    const = np.load(root / "flags" / "constant_run_24h.npy")
    valid = np.load(root / "masks" / POLICY / "valid_target.npy")
    year = np.load(root / "metadata" / "calendar" / "year.npy")
    sens = pd.read_csv(root / "metadata" / "sensors.csv")
    keep = ~const
    core.save_npy(d / "keep_mask.npy", keep)
    excl = valid & const
    r = dict(construction="keep_mask = NOT flags/constant_run_24h.npy (identical consecutive finite non-zero flow for >= 288 steps, "
                          "measured on the continuous timeline). Evaluation-only: AND-ed with the primary valid-target mask for Y/M; "
                          "windows, inputs, scalers and the primary mask are unchanged.",
             keep_mask_sha256=core.sha256_file(d / "keep_mask.npy"), primary_policy=POLICY,
             limitation="a run is flagged in full once it reaches 24 h; the first element of a repeated value is included, so the run "
                        "boundary is exact for the stored values but a genuinely steady reading cannot be told apart from a stuck one")
    by = {}
    for y in cfg["years"]:
        cy = year == y
        by[str(y)] = dict(flagged_steps=int(const[:, cy].sum()), excluded_valid_targets=int(excl[:, cy].sum()),
                          share_of_valid_targets=round(float(excl[:, cy].sum() / valid[:, cy].sum()), 6),
                          sensors_affected=int(const[:, cy].any(1).sum()),
                          sensors_by_geo_client={c: int(const[g][:, cy].any(1).sum()) for c, g in pipeline.groups(sens, "client_geo").items()},
                          sensors_by_iid_client={c: int(const[g][:, cy].any(1).sum()) for c, g in pipeline.groups(sens, "client_iid").items()})
    r["by_year"] = by
    csum = np.concatenate([np.zeros((200, 1), np.int64), np.cumsum(excl, 1, dtype=np.int64)], 1)
    vsum = np.concatenate([np.zeros((200, 1), np.int64), np.cumsum(valid, 1, dtype=np.int64)], 1)
    meta = json.loads((root / "experiments" / POLICY / "B" / "experiment.json").read_text())
    for split in ("val", "test"):
        w = np.load(root / "experiments" / POLICY / "B" / "windows" / f"{split}.npy").astype(np.int64)
        s, t = w[:, 0], w[:, 1] + meta["timeline_offset"]
        e = csum[s, t + 12] - csum[s, t]
        v = vsum[s, t + 12] - vsum[s, t]
        r[f"{split}_evaluation_population"] = dict(
            primary_windows=int(len(w)), primary_valid_targets=int(v.sum()),
            sensitivity_valid_targets=int((v - e).sum()), excluded_targets=int(e.sum()),
            windows_with_any_excluded_target=int((e > 0).sum()), windows_left_without_valid_targets=int(((v - e) == 0).sum()),
            same_for_A_and_B=True)
    core.save_json(d / "README.json", r)
    return r


# ---- I: smoke test ----

def smoke(cfg):
    import torch
    root = cfg["_out"]
    ds = ThreeYearWindows(root, POLICY, "B", "val", "pooled")
    X, Y, M, s = ds.get(arms.epoch_sample(len(ds), 1024, SEED, 5))
    tx, ty, tm = torch.from_numpy(X), torch.from_numpy(np.nan_to_num(ds.scale(Y, s))), torch.from_numpy(M)
    out = dict(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
               cpu=dict(x=list(tx.shape), dtype=str(tx.dtype), finite=bool(torch.isfinite(tx).all()),
                        masked_mean=float((ty * tm).sum() / tm.sum())))
    if torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            gx, gy, gm = tx.to(dev), ty.to(dev), tm.to(dev)
            w = torch.randn(12, 12, device=dev)
            pred = gx[..., 0] @ w
            loss = ((pred - gy).abs() * gm).sum() / gm.sum()
            torch.cuda.synchronize()
            out["cuda"] = dict(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                               total_mb=round(torch.cuda.get_device_properties(0).total_memory / 2**20),
                               x_on_gpu=str(gx.device), dtype=str(gx.dtype), finite=bool(torch.isfinite(loss).item()),
                               peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1))
        except Exception as e:                                      # record and continue
            out["cuda_error"] = repr(e)
    return out


# ---- J: manifests ----

def manifests(cfg):
    root = cfg["_out"]
    data_sha = core.sha256_file(root / "manifests" / "dgx_transfer_manifest.json")
    d = core.SRC / "aiori2" / "configs" / "experiments"
    written = []
    for arm in arms.ARMS:
        for h in arms.HISTORIES:
            m = arms.manifest(root, arm, h, POLICY,
                              dict(primary="none", evaluation_sensitivity="phase3/sensitivity/constant_run_24h/keep_mask.npy (reported separately)"),
                              SEED, cfg["_config_sha256"], data_sha, None)
            m["data_root"] = cfg["out_dir"]
            core.save_json(d / f"{arm}_{h}.json", m)
            written.append(f"aiori2/configs/experiments/{arm}_{h}.json")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2.validate")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--base-dir"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = core.load_config(None, a.base_dir, a.out)
    if a.smoke_only:
        print(json.dumps(smoke(cfg), indent=1))
        return
    P, rep, t0 = [], {}, time.time()
    for name, fn in (("B_sensors", lambda: sensors_and_partitions(cfg, P)), ("C_arms", lambda: arms_check(cfg, P)),
                     ("D_ab_leakage", lambda: ab_check(cfg, P)), ("EF_scalers_loader", lambda: loader_check(cfg, P)),
                     ("G_fairness", lambda: fairness(cfg)), ("H_sensitivity", lambda: sensitivity(cfg)),
                     ("I_smoke", lambda: smoke(cfg)), ("J_manifests", lambda: manifests(cfg))):
        t = time.time()
        rep[name] = fn()
        print(f"{name} done ({time.time() - t:.0f}s), problems so far: {len(P)}", flush=True)
    rep["problems"] = P
    rep["ok"] = not P
    rep["seconds"] = round(time.time() - t0, 1)
    core.save_json(cfg["_out"] / "phase3" / "phase3_validation.json", rep)
    print(json.dumps(dict(ok=rep["ok"], problems=P), indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
