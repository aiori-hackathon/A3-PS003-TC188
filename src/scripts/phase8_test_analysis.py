"""Phase 8 E4: pre-declared analysis of the one-time January test pass (docs/phase8_test_protocol.md sections 5-6).

Run from the project root, only after E3 has completed and E4 is approved:
    .venv\\Scripts\\python.exe scripts/phase8_test_analysis.py --protocol-sha256 <sha> --freeze-sha256 <sha>

Fixed before any test value exists: the S5 method (aiori/phase8_analysis.py, unchanged), contrasts C1-C4,
seed-averaged paired cells, sensor-cluster bootstrap within geographic clients (2,000 reps, seed 20261028),
freeway-direction cluster sensitivity, wider interval and larger p reported, Holm over C1-C4, SESOI 0.1638 taken
from the frozen validation log (never recomputed), and the wording rules with "on January test data".
Refuses to run unless the protocol, the freeze manifest, the frozen S5 outputs, this script (E4 record) and the
evaluator (E3 run log) match their hashes and E3 has completed. Writes only results/phase8/test/analysis_test.json
and summary_test.md (aggregate; refuses to overwrite).
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiori import config, phase8_analysis as pa, splits  # noqa: E402

ARMS = ("centralised", "local", "fedavg_iid", "fedavg_geo")
SEEDS = (0, 1, 2)
H = 12
SESOI_PROTOCOL = 0.1638
WORDING = {"supported on January validation data": "supported on January test data",
           "detectable but below the practical threshold": "detectable but below the practical threshold on January test data",
           "not distinguishable on this validation set": "not distinguishable on January test data"}
CATEGORY = {"supported on January validation data": "supported", "detectable but below the practical threshold": "detectable",
            "not distinguishable on this validation set": "not distinguishable"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR / "geo200")
    p.add_argument("--metadata", type=Path, default=config.PROCESSED_DIR / "sensor_metadata_aligned.csv")
    return p.parse_args(argv)


def gate(args):
    """Fail closed unless every frozen input and both scripts match their recorded hashes and E3 completed."""
    pr = args.root / "results" / "phase8"
    P = dict(protocol=args.root / "docs" / "phase8_test_protocol.md", freeze=pr / "test_protocol" / "frozen_checkpoints.json",
             e1=pr / "test_protocol" / "e1_record.json", e4=pr / "test_protocol" / "e4_record.json",
             sesoi=pr / "analysis" / "sesoi_log.json", s5=pr / "analysis" / "analysis.json", test=pr / "test",
             val_runs=pr / "geo200")
    if sha256_file(P["protocol"]) != args.protocol_sha256 or sha256_file(P["freeze"]) != args.freeze_sha256:
        raise SystemExit("Refusing: protocol or freeze manifest does not match the given hash")
    fz = json.loads(P["freeze"].read_text(encoding="utf-8"))
    e4 = json.loads(P["e4"].read_text(encoding="utf-8"))
    if e4["script_sha256"] != sha256_file(__file__) or (e4["protocol_sha256"], e4["freeze_sha256"]) != (
            args.protocol_sha256, args.freeze_sha256):
        raise SystemExit("Refusing: this analysis script or its record differs from the E4 freeze")
    deps = {args.root / k: v for k, v in e4["dependency_sha256"].items()}
    deps |= {P["sesoi"]: fz["sesoi_log_sha256"], P["s5"]: fz["analysis_json_sha256"]}
    bad = sorted(str(p) for p, h in deps.items() if not p.is_file() or sha256_file(p) != h)
    if bad:
        raise SystemExit(f"Refusing: frozen dependency changed: {bad[:3]}")
    if not (P["test"] / "RUN_COMPLETED").is_file():
        raise SystemExit("Refusing: E3 has not completed")
    if (P["test"] / "analysis_test.json").exists():
        raise SystemExit("Refusing to overwrite analysis_test.json")
    e1 = json.loads(P["e1"].read_text(encoding="utf-8"))
    log = json.loads((P["test"] / "run_log.json").read_text(encoding="utf-8"))
    if log["script_sha256"] != e1["script_sha256"] or log["windows_loaded"] != ["test"]:
        raise SystemExit("Refusing: E3 run log does not come from the frozen evaluator")
    sesoi = json.loads(P["sesoi"].read_text(encoding="utf-8"))["sesoi_mae"]
    if round(sesoi, 4) != SESOI_PROTOCOL:
        raise SystemExit(f"Refusing: SESOI {sesoi} differs from the protocol value {SESOI_PROTOCOL}")
    return P, fz, sesoi


def day_labels(steps, spd):
    """Calendar date and weekday of each day index (time convention of aiori.splits)."""
    out = {}
    for d in sorted(set(int(x) for x in steps)):
        ts = pd.Timestamp(splits.TIME_ORIGIN) + pd.Timedelta(days=d)
        out[d] = dict(date=f"{ts:%Y-%m-%d}", weekday=f"{ts:%a}", weekend=bool(ts.day_of_week >= 5))
    return out


def analyse(args, P, sesoi):
    man = pd.read_csv(args.data_dir / "sensor_selection.csv", usecols=["array_index", "proposed_client", "type_group", "flow_tertile"])
    meta = pd.read_csv(args.metadata, usecols=["Fwy", "Direction"])
    std = np.asarray(json.loads((args.data_dir / "scaler_parameters.json").read_text(encoding="utf-8"))["std"])
    spd = json.loads((args.data_dir / "split_config.json").read_text(encoding="utf-8"))["steps_per_day"]
    n_s = len(man)
    client = man["proposed_client"].to_numpy(str)
    fwy = meta.loc[man["array_index"], ["Fwy", "Direction"]].astype(str).agg("|".join, axis=1).to_numpy().astype(str)
    fwy_cluster = np.char.add(np.char.add(client, "|"), fwy)

    ref, per_run, err, stored, inputs = None, {}, {}, {}, {}
    for arm in ARMS:
        acc = {"abs": 0.0, "sq": 0.0, "z": 0.0}
        for s in SEEDS:
            d = P["test"] / f"{arm}_b1_geo200_seed{s}"
            f = next(d.glob("predictions_*_test.npz"))
            inputs[f"{d.name}/{f.name}"] = sha256_file(f)
            with np.load(f) as z:
                rec = {k: z[k] for k in ("sensor_pos", "t0", "mask", "y_hat", "y_true")}
            key = (rec["sensor_pos"], rec["t0"], rec["mask"])
            if ref is None:
                ref = key
            elif not all(np.array_equal(a, b) for a, b in zip(key, ref)):
                raise AssertionError(f"{arm} seed {s}: test cells differ between runs")
            m = json.loads((d / "metrics_test.json").read_text(encoding="utf-8"))["metrics"]
            e = np.where(rec["mask"], rec["y_hat"].astype(np.float64) - rec["y_true"].astype(np.float64), 0.0)
            if abs(np.abs(e).sum() / rec["mask"].sum() - m["pooled"]["mae"]) > 1e-6:
                raise AssertionError(f"{arm} seed {s}: pooled MAE disagrees with metrics_test.json")
            per_run[(arm, s)] = dict(abs=np.abs(e), neg=(rec["y_hat"] < 0) & rec["mask"])
            stored[(arm, s)] = m
            acc["abs"] = acc["abs"] + np.abs(e) / 3
            acc["sq"] = acc["sq"] + e ** 2 / 3
            acc["z"] = acc["z"] + np.abs(e) / std[rec["sensor_pos"]][:, None] / 3
        err[arm] = acc
    sp, t0, mask = ref
    step = t0[:, None] + np.arange(H)
    day_idx = step // spd
    days = day_labels(day_idx[mask], spd)
    dkeys = sorted(days)
    dmap = {d: i for i, d in enumerate(dkeys)}
    dkey = np.vectorize(dmap.get)(np.where(mask, day_idx, dkeys[0]))
    weekend_cell = np.vectorize(lambda d: days[d]["weekend"])(np.where(mask, day_idx, dkeys[0]))
    hkey = np.broadcast_to(np.arange(H), mask.shape)
    nd = len(dkeys)

    sums = lambda v, m=mask, key=None, k=1: pa.per_sensor_sums(v, m, sp, n_s, key, k)  # noqa: E731
    ones = np.ones(mask.shape)
    n, n_h, n_d = sums(ones), sums(ones, key=hkey, k=H), sums(ones, key=dkey, k=nd)
    wk = {"weekday": mask & ~weekend_cell, "weekend": mask & weekend_cell}
    n_w = {w: sums(ones, m) for w, m in wk.items()}
    S = {}
    for arm in ARMS:
        S[arm], S[arm + "|sq"], S[arm + "|z"] = sums(err[arm]["abs"]), sums(err[arm]["sq"]), sums(err[arm]["z"])
        for h, col in enumerate(sums(err[arm]["abs"], key=hkey, k=H).T):
            S[f"{arm}|h{h}"] = col
        S[f"{arm}|days"] = sums(err[arm]["abs"], key=dkey, k=nd)
        for w, m in wk.items():
            S[f"{arm}|{w}"] = sums(err[arm]["abs"], m)
    point = {arm: S[arm].sum() / n.sum() for arm in ARMS}

    # primary: identical to S5
    prim = {arm: S[arm] for arm in ARMS} | {"n": n}
    tb = pa.resample_totals(prim | {f"{a}|sq": S[a + "|sq"] for a in ARMS} | {f"{a}|z": S[a + "|z"] for a in ARMS}
                            | {f"{a}|h{h}": S[f"{a}|h{h}"] for a in ARMS for h in range(H)}
                            | {f"n|h{h}": n_h[:, h] for h in range(H)}
                            | {f"{a}|{w}": S[f"{a}|{w}"] for a in ARMS for w in wk} | {f"n|{w}": n_w[w] for w in wk},
                            client, np.arange(n_s))
    fb = pa.resample_totals(prim, client, fwy_cluster)
    s5 = json.loads(P["s5"].read_text(encoding="utf-8"))
    primary = {}
    for cid, (a, b) in pa.CONTRASTS.items():
        r = {"a": a, "b": b, "diff": point[a] - point[b]}
        for kind, t in (("sensor", tb), ("freeway", fb)):
            ci, p = pa.interval_and_p(pa.contrast(t[a], t[b], t["n"]))
            r[kind] = {"ci95": ci, "p": p}
        wider = max(("sensor", "freeway"), key=lambda k: np.diff(r[k]["ci95"])[0])
        r["reported_ci95"], r["reported_interval"] = r[wider]["ci95"], wider
        r["p_conservative"] = max(r["sensor"]["p"], r["freeway"]["p"])
        primary[cid] = r
    for cid, p in zip(primary, pa.holm([primary[c]["p_conservative"] for c in primary])):
        r = primary[cid]
        v = pa.verdict(r["reported_ci95"], p, r["diff"], sesoi)
        val = s5["primary"][cid]
        r.update(p_holm=p, abs_diff_ge_sesoi=abs(r["diff"]) >= sesoi, verdict=WORDING[v], category=CATEGORY[v],
                 validation=dict(diff=val["diff"], reported_ci95=val["reported_ci95"], p_holm=val["p_holm"],
                                 verdict=val["verdict"], category=CATEGORY[val["verdict"]]))
        r["consistent_with_validation"] = r["category"] == r["validation"]["category"]

    # exploratory (CIs only, no claims)
    def ci_of(ta, tb_, tn, kind="mae"):
        return pa.interval_and_p(pa.contrast(ta, tb_, tn, kind))[0]

    expl = {k: {} for k in ("rmse", "normalised_mae", "horizon_mae", "weekday_weekend_mae", "client_mae",
                            "type_group_mae", "tertile_mae")}
    for cid, (a, b) in pa.CONTRASTS.items():
        expl["rmse"][cid] = dict(diff=float(np.sqrt(S[a + "|sq"].sum() / n.sum()) - np.sqrt(S[b + "|sq"].sum() / n.sum())),
                                 ci95=ci_of(tb[a + "|sq"], tb[b + "|sq"], tb["n"], "rmse"))
        expl["normalised_mae"][cid] = dict(diff=float((S[a + "|z"].sum() - S[b + "|z"].sum()) / n.sum()),
                                           ci95=ci_of(tb[a + "|z"], tb[b + "|z"], tb["n"]))
        expl["horizon_mae"][cid] = [dict(minutes=5 * (h + 1), diff=float((S[f"{a}|h{h}"].sum() - S[f"{b}|h{h}"].sum()) / n_h[:, h].sum()),
                                         ci95=ci_of(tb[f"{a}|h{h}"], tb[f"{b}|h{h}"], tb[f"n|h{h}"])) for h in range(H)]
        expl["weekday_weekend_mae"][cid] = {w: dict(diff=float((S[f"{a}|{w}"].sum() - S[f"{b}|{w}"].sum()) / n_w[w].sum()),
                                                    ci95=ci_of(tb[f"{a}|{w}"], tb[f"{b}|{w}"], tb[f"n|{w}"])) for w in wk}
        for label, groups in (("client_mae", client), ("type_group_mae", man["type_group"].to_numpy(str)),
                              ("tertile_mae", man["flow_tertile"].to_numpy(str))):
            expl[label][cid] = {}
            for g in sorted(set(groups)):
                m = groups == g
                t = pa.resample_totals({"a": S[a][m], "b": S[b][m], "n": n[m]}, client[m], np.arange(m.sum()))
                expl[label][cid][g] = dict(n_sensors=int(m.sum()), diff=float((S[a][m].sum() - S[b][m].sum()) / n[m].sum()),
                                           ci95=ci_of(t["a"], t["b"], t["n"]))

    # sensitivity: leave one test day out (points), per day, matched seeds 1-2
    lodo = {days[dkeys[i]]["date"]: {cid: float((np.delete(S[f"{a}|days"], i, 1).sum() - np.delete(S[f"{b}|days"], i, 1).sum())
                                                / np.delete(n_d, i, 1).sum()) for cid, (a, b) in pa.CONTRASTS.items()}
            for i in range(nd)}
    per_day_contrast = {days[dkeys[i]]["date"]: {cid: float((S[f"{a}|days"][:, i].sum() - S[f"{b}|days"][:, i].sum()) / n_d[:, i].sum())
                                                 for cid, (a, b) in pa.CONTRASTS.items()} for i in range(nd)}
    no_seed0 = {cid: float(np.mean([per_run[(a, s)]["abs"].sum() - per_run[(b, s)]["abs"].sum() for s in (1, 2)]) / mask.sum())
                for cid, (a, b) in pa.CONTRASTS.items()}

    # descriptive tables
    sensor_mae = {arm: S[arm] / n for arm in ARMS}
    neg = {arm: dict(overall=float(np.mean([per_run[(arm, s)]["neg"].sum() / mask.sum() for s in SEEDS])),
                     weekend=float(np.mean([per_run[(arm, s)]["neg"][wk["weekend"]].sum() / wk["weekend"].sum() for s in SEEDS]))
                     if wk["weekend"].any() else None) for arm in ARMS}
    ref_fc = json.loads((P["test"] / "reference_forecasts_test.json").read_text(encoding="utf-8"))
    val_point = s5["descriptive"]["pooled_mae_seed_avg"]
    desc = dict(
        pooled_mae=point,
        pooled_rmse={a: float(np.sqrt(S[a + "|sq"].sum() / n.sum())) for a in ARMS},
        normalised_mae={a: float(S[a + "|z"].sum() / n.sum()) for a in ARMS},
        sensor_macro_mae={a: float(np.mean(sensor_mae[a])) for a in ARMS},
        per_seed={f"{a}|{s}": dict(mae=stored[(a, s)]["pooled"]["mae"], rmse=stored[(a, s)]["pooled"]["rmse"],
                                   sensor_macro_mae=stored[(a, s)]["sensor_macro"]["mae"],
                                   normalised_mae=stored[(a, s)]["normalised"]["mae"]) for a in ARMS for s in SEEDS},
        per_seed_contrasts={cid: [float((per_run[(a, s)]["abs"].sum() - per_run[(b, s)]["abs"].sum()) / mask.sum()) for s in SEEDS]
                            for cid, (a, b) in pa.CONTRASTS.items()},
        client={a: {c: dict(mae=float(S[a][client == c].sum() / n[client == c].sum()),
                            rmse=float(np.sqrt(S[a + "|sq"][client == c].sum() / n[client == c].sum())),
                            macro_mae=float(np.mean(sensor_mae[a][client == c]))) for c in sorted(set(client))} for a in ARMS},
        horizon_mae={a: [float(S[f"{a}|h{h}"].sum() / n_h[:, h].sum()) for h in range(H)] for a in ARMS},
        day_mae={a: {days[dkeys[i]]["date"]: float(S[f"{a}|days"][:, i].sum() / n_d[:, i].sum()) for i in range(nd)} for a in ARMS},
        weekday_weekend_mae={a: {w: float(S[f"{a}|{w}"].sum() / n_w[w].sum()) for w in wk} for a in ARMS},
        days=days,
        sensor_mae_distribution={a: dict(median=float(np.median(sensor_mae[a])), iqr=np.percentile(sensor_mae[a], [25, 75]).tolist(),
                                         p5_p95=np.percentile(sensor_mae[a], [5, 95]).tolist()) for a in ARMS},
        sensor_level={cid: dict(a_better=int((sensor_mae[a] < sensor_mae[b]).sum()), b_better=int((sensor_mae[a] > sensor_mae[b]).sum()),
                                median_diff=float(np.median(sensor_mae[a] - sensor_mae[b]))) for cid, (a, b) in pa.CONTRASTS.items()},
        negative_prediction_rate=neg,
        reference_forecasts={k: dict(mae=v["pooled"]["mae"], rmse=v["pooled"]["rmse"],
                                     client_mae={c: x["mae"] for c, x in v["by_client"].items()},
                                     horizon_mae=[x["mae"] for x in v["by_horizon"]]) for k, v in ref_fc.items()},
        arm_minus_reference={a: {k: point[a] - v["pooled"]["mae"] for k, v in ref_fc.items()} for a in ARMS},
        validation_vs_test={a: dict(validation_mae=val_point[a], test_mae=point[a], change=point[a] - val_point[a],
                                    ratio=point[a] / val_point[a]) for a in ARMS},
    )
    return dict(primary=primary, exploratory=expl, descriptive=desc, leave_one_day_out=lodo, per_day_contrasts=per_day_contrast,
                matched_seeds_without_seed0=no_seed0, sesoi=sesoi,
                design=dict(n_sensors=n_s, n_cells=int(mask.sum()), n_windows=int(len(t0)), n_days=nd,
                            n_freeway_clusters=int(len(set(fwy_cluster))), reps=pa.BOOT_REPS, seed=pa.BOOT_SEED,
                            strata="geographic client", id_mapping_status=config.ID_MAPPING_STATUS),
                input_sha256=inputs)


def summary_markdown(r):
    """Aggregate summary tables generated mechanically (no sensor IDs)."""
    d = r["descriptive"]
    lines = ["# Phase 8 January test results (generated summary; aggregate only)", "",
             "| Arm | Test MAE | Test RMSE | Normalised MAE | Validation MAE | Change | Negative predictions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        v = d["validation_vs_test"][a]
        lines.append(f"| {a} | {d['pooled_mae'][a]:.3f} | {d['pooled_rmse'][a]:.3f} | {d['normalised_mae'][a]:.4f} | "
                     f"{v['validation_mae']:.3f} | {v['change']:+.3f} | {100 * d['negative_prediction_rate'][a]['overall']:.2f}% |")
    lines += ["", "| Reference forecast | Test MAE | Test RMSE |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['mae']:.3f} | {v['rmse']:.3f} |" for k, v in d["reference_forecasts"].items()]
    lines += ["", f"SESOI = {r['sesoi']:.4f} (fixed from validation).", "",
              "| | Δ | Reported 95% CI | Holm p | Verdict (test) | Validation verdict | Consistent |", "|---|---:|---|---:|---|---|---|"]
    for cid, p in r["primary"].items():
        lines.append(f"| {cid} {p['a']} − {p['b']} | {p['diff']:+.3f} | [{p['reported_ci95'][0]:.3f}, {p['reported_ci95'][1]:.3f}] | "
                     f"{p['p_holm']:.4f} | {p['verdict']} | {p['validation']['verdict']} | {p['consistent_with_validation']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    P, fz, sesoi = gate(args)
    res = analyse(args, P, sesoi)
    res["created_utc"] = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}"
    res["script_sha256"] = sha256_file(__file__)
    res["protocol_sha256"], res["freeze_sha256"] = args.protocol_sha256, args.freeze_sha256
    (P["test"] / "analysis_test.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (P["test"] / "summary_test.md").write_text(summary_markdown(res), encoding="utf-8")
    print(f"E4 analysis written to {P['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
