"""Phase 8 S5: pre-declared analysis of the 12 saved validation prediction records.

Run from the project root (after the SESOI has been recorded in results/phase8/analysis/sesoi_log.json):
    .venv\\Scripts\\python.exe scripts/phase8_analysis.py

Reads only the saved validation predictions, metrics.json files and the geo200 manifest/scaler/metadata;
no model is loaded or trained, and the test split is never read. Writes only
results/phase8/analysis/analysis.json (aggregate statistics; refuses to overwrite).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiori import config, phase8_analysis as pa  # noqa: E402

RUNS = ROOT / "results" / "phase8" / "geo200"
OUT = ROOT / "results" / "phase8" / "analysis"
GEO = config.PROCESSED_DIR / "geo200"
ARMS = ("centralised", "local", "fedavg_iid", "fedavg_geo")
SEEDS = (0, 1, 2)
H = 12


def load(arm, seed):
    d = RUNS / f"{arm}_b1_geo200_seed{seed}"
    with np.load(next(d.glob("predictions_*_val.npz"))) as z:
        rec = {k: z[k] for k in ("sensor_pos", "t0", "y_hat", "y_true", "mask")}
    stored = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return rec, stored


def main():
    out_file = OUT / "analysis.json"
    if out_file.exists():
        raise SystemExit(f"Refusing to overwrite {out_file}")
    log = json.loads((OUT / "sesoi_log.json").read_text(encoding="utf-8"))
    sesoi = log["sesoi_mae"]

    man = pd.read_csv(GEO / "sensor_selection.csv", usecols=["array_index", "proposed_client", "type_group", "flow_tertile"])
    meta = pd.read_csv(config.PROCESSED_DIR / "sensor_metadata_aligned.csv", usecols=["Fwy", "Direction"])
    std = np.asarray(json.loads((GEO / "scaler_parameters.json").read_text(encoding="utf-8"))["std"])
    split = json.loads((GEO / "split_config.json").read_text(encoding="utf-8"))
    val0, spd = split["boundaries"]["val"][0], split["steps_per_day"]
    n_s = len(man)
    client = man["proposed_client"].to_numpy(str)
    fwy = meta.loc[man["array_index"], ["Fwy", "Direction"]].astype(str).agg("|".join, axis=1).to_numpy()
    fwy_cluster = np.char.add(np.char.add(client, "|"), fwy.astype(str))

    # per-cell absolute / squared / normalised errors, averaged over seeds, paired across arms
    ref, err, per_run, history = None, {}, {}, {}
    for arm in ARMS:
        acc = {"abs": 0.0, "sq": 0.0}
        for s in SEEDS:
            rec, stored = load(arm, s)
            key = (rec["sensor_pos"], rec["t0"], rec["mask"])
            if ref is None:
                ref = key
            elif not all(np.array_equal(a, b) for a, b in zip(key, ref)):
                raise AssertionError(f"{arm} seed {s}: validation cells differ")
            e = np.where(rec["mask"], rec["y_hat"].astype(np.float64) - rec["y_true"].astype(np.float64), 0.0)
            mae = np.abs(e).sum() / rec["mask"].sum()
            if abs(mae - stored["metrics"]["pooled"]["mae"]) > 1e-6:
                raise AssertionError(f"{arm} seed {s}: pooled MAE {mae} != stored {stored['metrics']['pooled']['mae']}")
            per_run[(arm, s)] = np.abs(e)
            acc["abs"] = acc["abs"] + np.abs(e) / len(SEEDS)
            acc["sq"] = acc["sq"] + e ** 2 / len(SEEDS)
            history[(arm, s)] = stored
        err[arm] = acc
    sp, t0, mask = ref
    hkey = np.broadcast_to(np.arange(H), mask.shape)
    dkey = (t0[:, None] + np.arange(H) - val0) // spd
    assert dkey[mask].min() >= 0 and dkey[mask].max() <= 2
    n_days = 3
    sums = lambda v, key=None, k=1: pa.per_sensor_sums(v, mask, sp, n_s, key, k)  # noqa: E731
    ones = np.ones(mask.shape)
    n = sums(ones)
    n_h, n_d = sums(ones, hkey, H), sums(ones, dkey, n_days)
    S = {}
    for arm in ARMS:
        S[arm] = sums(err[arm]["abs"])
        S[arm + "|sq"] = sums(err[arm]["sq"])
        S[arm + "|z"] = sums(err[arm]["abs"] / std[sp][:, None])
        for h, col in enumerate(sums(err[arm]["abs"], hkey, H).T):
            S[f"{arm}|h{h}"] = col
        S[f"{arm}|days"] = sums(err[arm]["abs"], dkey, n_days)
    point = {arm: S[arm].sum() / n.sum() for arm in ARMS}

    # primary: C1-C4 on pooled MAE, sensor-cluster and freeway-direction-cluster bootstraps within clients
    prim = {arm: S[arm] for arm in ARMS} | {"n": n}
    boots = {"sensor": pa.resample_totals(prim | {f"{a}|sq": S[a + "|sq"] for a in ARMS} | {f"{a}|z": S[a + "|z"] for a in ARMS}
                                          | {f"{a}|h{h}": S[f"{a}|h{h}"] for a in ARMS for h in range(H)}
                                          | {f"n|h{h}": n_h[:, h] for h in range(H)}, client, np.arange(n_s)),
             "freeway": pa.resample_totals(prim, client, fwy_cluster)}
    primary = {}
    for cid, (a, b) in pa.CONTRASTS.items():
        diff = point[a] - point[b]
        res = {"a": a, "b": b, "diff": diff}
        for kind, t in boots.items():
            ci, p = pa.interval_and_p(pa.contrast(t[a], t[b], t["n"]))
            res[kind] = {"ci95": ci, "p": p}
        wider = max(("sensor", "freeway"), key=lambda k: np.diff(res[k]["ci95"])[0])
        res["reported_ci95"], res["reported_interval"] = res[wider]["ci95"], wider
        res["p_conservative"] = max(res["sensor"]["p"], res["freeway"]["p"])
        primary[cid] = res
    for cid, p in zip(primary, pa.holm([primary[c]["p_conservative"] for c in primary])):
        r = primary[cid]
        r["p_holm"] = p
        r["abs_diff_ge_sesoi"] = abs(r["diff"]) >= sesoi
        r["verdict"] = pa.verdict(r["reported_ci95"], p, r["diff"], sesoi)

    # exploratory (CIs only, no claims): RMSE, normalised MAE, per horizon, per client, per type group / tertile
    tb = boots["sensor"]
    expl = {"rmse": {}, "normalised_mae": {}, "horizon_mae": {}, "client_mae": {}, "type_group_mae": {}, "tertile_mae": {}}
    for cid, (a, b) in pa.CONTRASTS.items():
        expl["rmse"][cid] = dict(diff=float(np.sqrt(S[a + "|sq"].sum() / n.sum()) - np.sqrt(S[b + "|sq"].sum() / n.sum())),
                                 ci95=pa.interval_and_p(pa.contrast(tb[a + "|sq"], tb[b + "|sq"], tb["n"], "rmse"))[0])
        expl["normalised_mae"][cid] = dict(diff=float((S[a + "|z"].sum() - S[b + "|z"].sum()) / n.sum()),
                                           ci95=pa.interval_and_p(pa.contrast(tb[a + "|z"], tb[b + "|z"], tb["n"]))[0])
        expl["horizon_mae"][cid] = [dict(minutes=5 * (h + 1), diff=float((S[f"{a}|h{h}"].sum() - S[f"{b}|h{h}"].sum()) / n_h[:, h].sum()),
                                         ci95=pa.interval_and_p(pa.contrast(tb[f"{a}|h{h}"], tb[f"{b}|h{h}"], tb[f"n|h{h}"]))[0])
                                    for h in range(H)]
        for label, groups in (("client_mae", client), ("type_group_mae", man["type_group"].to_numpy(str)),
                              ("tertile_mae", man["flow_tertile"].to_numpy(str))):
            expl[label][cid] = {}
            for g in sorted(set(groups)):
                m = groups == g
                t = pa.resample_totals({"a": S[a][m], "b": S[b][m], "n": n[m]}, client[m], np.arange(m.sum()))
                expl[label][cid][g] = dict(n_sensors=int(m.sum()), diff=float((S[a][m].sum() - S[b][m].sum()) / n[m].sum()),
                                           ci95=pa.interval_and_p(pa.contrast(t["a"], t["b"], t["n"]))[0])

    # descriptive: per arm, per seed, per client, per horizon, per sensor
    sensor_mae = {arm: S[arm] / n for arm in ARMS}
    desc = {
        "pooled_mae_seed_avg": point,
        "per_seed_pooled_mae": {f"{a}|{s}": float(per_run[(a, s)].sum() / mask.sum()) for a in ARMS for s in SEEDS},
        "per_seed_contrasts": {cid: [float(per_run[(a, s)].sum() / mask.sum() - per_run[(b, s)].sum() / mask.sum()) for s in SEEDS]
                               for cid, (a, b) in pa.CONTRASTS.items()},
        "client_mae": {arm: {c: float(S[arm][client == c].sum() / n[client == c].sum()) for c in sorted(set(client))} for arm in ARMS},
        "horizon_mae": {arm: [float(S[f"{arm}|h{h}"].sum() / n_h[:, h].sum()) for h in range(H)] for arm in ARMS},
        "sensor_level": {cid: dict(a_better=int((sensor_mae[a] < sensor_mae[b]).sum()), b_better=int((sensor_mae[a] > sensor_mae[b]).sum()),
                                   median_diff=float(np.median(sensor_mae[a] - sensor_mae[b])),
                                   p5_p95=np.percentile(sensor_mae[a] - sensor_mae[b], [5, 95]).tolist())
                         for cid, (a, b) in pa.CONTRASTS.items()},
        "sensor_mae_distribution": {arm: dict(median=float(np.median(sensor_mae[arm])),
                                              iqr=np.percentile(sensor_mae[arm], [25, 75]).tolist(),
                                              p5_p95=np.percentile(sensor_mae[arm], [5, 95]).tolist()) for arm in ARMS},
    }
    # leave-one-validation-day-out (point estimates; 3 days give stability, not an interval)
    lodo = {}
    for drop in range(n_days):
        keep = [d for d in range(n_days) if d != drop]
        nn = n_d[:, keep].sum()
        lodo[f"drop_day{drop + 1}"] = {cid: float((S[f"{a}|days"][:, keep].sum() - S[f"{b}|days"][:, keep].sum()) / nn)
                                       for cid, (a, b) in pa.CONTRASTS.items()}
    per_day = {f"day{d + 1}": {cid: float((S[f"{a}|days"][:, d].sum() - S[f"{b}|days"][:, d].sum()) / n_d[:, d].sum())
                               for cid, (a, b) in pa.CONTRASTS.items()} for d in range(n_days)}
    # matched-seed sensitivity without seed 0 (descriptive)
    no_seed0 = {cid: float(np.mean([per_run[(a, s)].sum() - per_run[(b, s)].sum() for s in (1, 2)]) / mask.sum())
                for cid, (a, b) in pa.CONTRASTS.items()}

    # geographic FedAvg seed 0 hit the 50-round cap: trajectory from the saved history only
    def traj(arm, s):
        st = history[(arm, s)]
        h = [x["val_mae"] for x in st["history"]]
        best = int(np.argmin(h)) + 1
        return dict(rounds_run=len(h), best_round=best, best_val_mae=min(h), early_stopped=st["summary"]["early_stopped"],
                    first_round_within_1pct=int(next(i + 1 for i, v in enumerate(h) if v <= 1.01 * min(h))),
                    first_round_within_0_5pct=int(next(i + 1 for i, v in enumerate(h) if v <= 1.005 * min(h))),
                    rounds_after_best=len(h) - best, gain_last_10_rounds_to_best=float(h[max(0, best - 11)] - min(h)),
                    val_mae_at_rounds={r: float(h[r - 1]) for r in (10, 20, 23, 28, 36, 40, 43) if r <= len(h)},
                    tail=[round(v, 4) for v in h[-8:]])
    cap = {f"{a}|{s}": traj(a, s) for a in ("fedavg_geo", "fedavg_iid") for s in SEEDS}

    result = dict(created_utc=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}", sesoi=log, primary=primary, exploratory=expl,
                  descriptive=desc, leave_one_day_out=lodo, per_day=per_day, matched_seeds_without_seed0=no_seed0,
                  round_cap_investigation=cap,
                  design=dict(n_sensors=n_s, n_cells=int(mask.sum()), n_windows=int(len(t0)), n_freeway_clusters=int(len(set(fwy_cluster))),
                              freeway_cluster_sizes=dict(zip(["min", "median", "max"],
                                                             np.percentile(pd.Series(fwy_cluster).value_counts(), [0, 50, 100]).tolist())),
                              reps=pa.BOOT_REPS, seed=pa.BOOT_SEED, strata="geographic client", seeds=list(SEEDS),
                              id_mapping_status=config.ID_MAPPING_STATUS))
    OUT.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
