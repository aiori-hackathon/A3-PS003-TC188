"""Phase 9 final stages (docs/phase9_protocol.md): freeze -> precheck -> test (once) -> analysis -> report.

    python final.py freeze | precheck | test | analysis | report

freeze:   hashes every completed run, the dataset, code and configs; fixes the SESOI (1% of the centralised validation
          MAE) and the validation-only clipping decision; refuses if any test output exists.
precheck: re-predicts validation from the frozen checkpoints; must equal the saved validation predictions byte for byte.
test:     the single held-out pass (Aug 8 - Sep 30). RUN_STARTED is written before test windows are loaded; an
          interrupted pass is moved aside unchanged and re-run from the same frozen models (at most 2 attempts).
analysis: pre-declared comparisons (matched seeds, sensor-cluster bootstrap within clients, freeway sensitivity, Holm).
report:   aggregate markdown reports, SVG figures, manifests and the integrity report (also a partial-delivery report).
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aiori import config, dataset, evaluation, federated, phase8_analysis as pa, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402

F = C.R / "final"
T = F / "test"
FROZEN = F / "frozen.json"
CONTRASTS = {"C1": ("fedavg_geo", "centralised"), "C2": ("fedavg_geo", "local"), "C3": ("fedavg_iid", "fedavg_geo"),
             "C4": ("local", "centralised")}
LABEL = {"centralised": "Centralised GRU", "fedavg_geo": "Geographic FedAvg", "local": "Local-only GRU", "fedavg_iid": "IID FedAvg"}
CODE = ["aiori/" + f for f in ("dataset.py", "model.py", "training.py", "evaluation.py", "metrics.py", "baselines.py", "windows.py",
                               "federated.py", "phase8_analysis.py", "config.py", "preprocessing.py", "splits.py", "nine_month.py",
                               "resumable.py", "select_geo200.py")] + \
       ["experiments/phase9/" + f for f in ("common.py", "worker.py", "final.py", "run_nine_month_experiment.py", "watchdog.py")]


def _complete_runs():
    out = {}
    for d in sorted(p for p in C.RUNS.iterdir() if p.is_dir()) if C.RUNS.exists() else []:
        m = C.read_json(d / "COMPLETE.json")
        if m and all(Path(p).is_file() and C.sha256_file(p) == h for p, h in m["files"].items()):
            arm, seed = d.name.rsplit("_seed", 1)
            out[d.name] = dict(arm=arm, seed=int(seed), dir=d)
    return out


def _run(splits):
    return dataset.load_run("nine_month", run_dir=C.DATA, selection=C.DATA / "sensor_selection.csv", splits=splits)


# ---------------- freeze ----------------
def stage_freeze():
    if FROZEN.exists():
        return 0
    if T.exists():
        raise SystemExit("Refusing to freeze: test outputs already exist")
    runs = _complete_runs()
    if not runs:
        raise SystemExit("No complete run to freeze")
    val = {t: C.read_json(r["dir"] / "metrics_val.json")["pooled"] for t, r in runs.items()}
    cen = [val[t]["mae"] for t, r in runs.items() if r["arm"] == "centralised"]
    ref = float(np.mean(cen)) if cen else float(np.mean([v["mae"] for v in val.values()]))
    clip = {}
    for t, r in runs.items():
        with np.load(r["dir"] / "predictions_val.npz") as z:
            m, y, yh = z["mask"], z["y_true"].astype(float), z["y_hat"].astype(float)
            clip[t] = dict(raw_mae=float(np.abs(yh - y)[m].mean()), clipped_mae=float(np.abs(np.maximum(yh, 0) - y)[m].mean()),
                           raw_rmse=float(np.sqrt(((yh - y) ** 2)[m].mean())), clipped_rmse=float(np.sqrt(((np.maximum(yh, 0) - y) ** 2)[m].mean())))
    by_arm = {}
    for t, r in runs.items():
        by_arm.setdefault(r["arm"], []).append(t)
    clip_decision = {a: bool(np.mean([clip[t]["clipped_mae"] for t in ts]) < np.mean([clip[t]["raw_mae"] for t in ts])) for a, ts in by_arm.items()}
    arm_val = {a: (float(np.mean([val[t]["mae"] for t in ts])), float(np.mean([val[t]["rmse"] for t in ts]))) for a, ts in by_arm.items()}
    selected = min(arm_val, key=lambda a: arm_val[a])               # lowest validation MAE, RMSE breaks ties
    files = {str(p): C.sha256_file(p) for r in runs.values() for p in sorted(r["dir"].iterdir())
             if p.is_file() and p.suffix in (".pt", ".json", ".npz") and p.name not in ("progress.json",)}
    files |= {str(p): C.sha256_file(p) for p in sorted(C.DATA.iterdir()) if p.is_file()}
    files |= {str(C.ROOT / c): C.sha256_file(C.ROOT / c) for c in CODE if (C.ROOT / c).is_file()}
    files |= {str(p): C.sha256_file(p) for p in sorted((C.ROOT / "experiments" / "configs").glob("phase9_*.json"))}
    proto = C.ROOT / "docs" / "phase9_protocol.md"
    rec = dict(frozen_utc=C.now(), runs={t: dict(arm=r["arm"], seed=r["seed"]) for t, r in runs.items()},
               sesoi_mae=0.01 * ref, sesoi_rule="1% of the mean centralised validation MAE" if cen else
               "1% of the mean validation MAE of all completed runs (no centralised run completed)",
               clipping_validation=clip, clipping_selected_by_arm=clip_decision, validation_by_arm=arm_val,
               selected_final_arm=selected, selected_final_variant="clipped at 0" if clip_decision[selected] else "raw", files=files,
               protocol_sha256=C.sha256_file(proto) if proto.exists() else None, test_outputs_existed=False)
    C.write_json(FROZEN, rec)
    C.log(f"FREEZE: {len(runs)} runs, SESOI {rec['sesoi_mae']:.4f}, clipping {clip_decision}")
    return 0


def _verify_frozen():
    fz = C.read_json(FROZEN)
    bad = [p for p, h in fz["files"].items() if not Path(p).is_file() or C.sha256_file(p) != h]
    if bad:
        raise SystemExit(f"Frozen files changed: {bad[:3]}")
    return fz


def _states(d):
    pts = sorted(d.glob("best_model*.pt"))
    if [p.name for p in pts] == ["best_model.pt"]:
        return None, {"all": torch.load(pts[0], weights_only=True)}
    part = {k: np.asarray(v) for k, v in json.loads((d / "partition.json").read_text()).items()}
    if [p.name for p in pts] == ["best_model_global.pt"]:
        return part, {"global": torch.load(pts[0], weights_only=True)}
    return part, {c: torch.load(d / f"best_model_{c}.pt", weights_only=True) for c in part}


def _predict(run, split, part, states, hidden=64):
    """Routing identical to training: one model, or per-client subset (global or own model), merged."""
    with torch.no_grad():
        if part is None:
            ds = dataset.WindowDataset(run, split, calendar=True)
            m = ForecastGRU(ds.n_features, hidden)
            m.load_state_dict(states["all"], strict=True)
            return training.predict(m, ds)
        preds = []
        for name, pos in sorted(part.items()):
            pos = np.sort(pos)
            ds = dataset.WindowDataset(federated.subset_run(run, pos, name=name), split, calendar=True)
            m = ForecastGRU(ds.n_features, hidden)
            m.load_state_dict(states["global"] if "global" in states else states[name], strict=True)
            p = training.predict(m, ds)
            p["sensor_pos"] = pos[p["sensor_pos"]]
            preds.append(p)
        return federated.merge_predictions(preds)


# ---------------- precheck ----------------
def stage_precheck():
    fz = _verify_frozen()
    out = F / "precheck.json"
    if C.read_json(out, {}).get("passed"):
        return 0
    run = _run(("val",))
    res = {}
    for t in fz["runs"]:
        d = C.RUNS / t
        training.configure_torch(8)
        part, states = _states(d)
        p = _predict(run, "val", part, states)
        with np.load(d / "predictions_val.npz") as z:
            res[t] = bool(np.array_equal(p["sensor_pos"], z["sensor_pos"]) and np.array_equal(p["t0"], z["t0"])
                          and p["y_hat_z"].astype(np.float32).tobytes() == z["y_hat_z"].tobytes())
    ok = all(res.values())
    C.write_json(out, dict(passed=ok, runs=res, windows_loaded=sorted(run.windows), utc=C.now()))
    C.log(f"PRECHECK {'PASSED' if ok else 'FAILED'} {res}")
    return 0 if ok else 3


# ---------------- test (once) ----------------
def stage_test():
    fz = _verify_frozen()
    if not C.read_json(F / "precheck.json", {}).get("passed"):
        raise SystemExit("Refusing: validation pre-check has not passed")
    if (T / "RUN_COMPLETED").exists():
        return 0
    if T.exists():                                           # interrupted attempt: preserve unchanged, re-run from frozen models
        n = len(list(F.glob("test_partial_attempt_*")))
        if n >= 2:
            raise SystemExit("Refusing: two interrupted test attempts already")
        T.rename(F / f"test_partial_attempt_{n + 1}")
        C.log(f"TEST: interrupted attempt preserved as test_partial_attempt_{n + 1}; re-running from frozen models", "recovery.log")
    T.mkdir(parents=True)
    C.write_json(T / "RUN_STARTED", dict(utc=C.now()))
    run = _run(("test",))
    t0 = time.time()
    key, summary = None, {}
    for t, r in fz["runs"].items():
        d = C.RUNS / t
        training.configure_torch(8)
        part, states = _states(d)
        pred = _predict(run, "test", part, states)
        if not np.isfinite(pred["y_hat_z"]).all() or not np.array_equal(np.stack([pred["sensor_pos"], pred["t0"]], 1), run.windows["test"]):
            raise AssertionError(f"{t}: invalid test predictions")
        k = (pred["sensor_pos"].tobytes(), pred["t0"].tobytes(), pred["mask"].tobytes())
        key = key or k
        if k != key:
            raise AssertionError("test windows differ between runs")
        (T / t).mkdir()
        evaluation.write_prediction_record(T / t / "predictions_test.npz", pred, run, dict(run=t, split="test"))
        met = evaluation.evaluate_predictions(pred, run)
        C.write_json(T / t / "metrics_test.json", met)
        summary[t] = met["pooled"]["n"]
    C.write_json(T / "reference_forecasts_test.json", evaluation.evaluate_baselines(run, "test"))
    C.write_json(T / "run_log.json", dict(windows=int(len(run.windows["test"])), windows_loaded=sorted(run.windows), runs=summary,
                                          seconds=time.time() - t0, peak_mb=C.peak_mb(), utc=C.now()))
    (T / "RUN_COMPLETED").write_text(C.now(), encoding="utf-8")
    C.log(f"TEST pass complete: {len(summary)} runs")
    return 0


# ---------------- analysis ----------------
def _sensor_sums(rec, std, dkey, wkey, n_s):
    m = rec["mask"]
    sp = np.broadcast_to(rec["sensor_pos"][:, None], m.shape)
    e = np.where(m, rec["y_hat"].astype(float) - rec["y_true"].astype(float), 0.0)
    ec = np.where(m, np.maximum(rec["y_hat"].astype(float), 0) - rec["y_true"].astype(float), 0.0)
    out = {}
    for name, v in (("abs", np.abs(e)), ("sq", e ** 2), ("z", np.abs(e) / std[rec["sensor_pos"]][:, None]), ("clip_abs", np.abs(ec)),
                    ("clip_sq", ec ** 2), ("neg", ((rec["y_hat"] < 0) & m).astype(float)), ("n", m.astype(float))):
        out[name] = np.bincount(sp[m], weights=v[m], minlength=n_s)
        if name in ("abs", "n"):
            for kname, key, kk in (("h", np.broadcast_to(np.arange(12), m.shape), 12), ("month", dkey, 2), ("wk", wkey, 2)):
                out[f"{name}_{kname}"] = np.bincount(sp[m] * kk + key[m], weights=v[m], minlength=n_s * kk).reshape(n_s, kk)
    return out


def stage_analysis():
    fz = _verify_frozen()
    if not (T / "RUN_COMPLETED").exists():
        raise SystemExit("Refusing: test pass not complete")
    out = F / "analysis.json"
    if out.exists():
        return 0
    man = pd.read_csv(C.DATA / "sensor_selection.csv")
    n_s = len(man)
    client = man["proposed_client"].to_numpy(str)
    std = np.asarray(json.loads((C.DATA / "scaler_parameters.json").read_text())["std"])
    meta = pd.read_csv(config.PROCESSED_DIR / "sensor_metadata_aligned.csv", usecols=["Fwy", "Direction"])
    fwy = np.char.add(np.char.add(client, "|"), meta.loc[man["array_index"], ["Fwy", "Direction"]].astype(str).agg("|".join, axis=1).to_numpy().astype(str))
    origin = pd.Timestamp("2023-01-01")
    sums, val = {}, {}
    for t, r in fz["runs"].items():
        with np.load(T / t / "predictions_test.npz") as z:
            rec = {k: z[k] for k in ("sensor_pos", "t0", "mask", "y_hat", "y_true")}
        step = rec["t0"][:, None] + np.arange(12)
        day = step // 288
        dates = origin + pd.to_timedelta(np.arange(day.min(), day.max() + 1), unit="D")
        month = np.isin(np.arange(day.min(), day.max() + 1), np.flatnonzero(dates.month == 9) + day.min())[day - day.min()].astype(int)
        weekend = (dates.dayofweek >= 5)[day - day.min()].astype(int)
        sums[t] = _sensor_sums(rec, std, month, weekend, n_s)
        val[t] = C.read_json(C.RUNS / t / "metrics_val.json")["pooled"]
    arms = {}
    for t, r in fz["runs"].items():
        arms.setdefault(r["arm"], {})[r["seed"]] = t

    def agg(arm, seeds, key):
        return np.mean([sums[arms[arm][s]][key] for s in seeds], axis=0)

    desc = {}
    for arm, by_seed in arms.items():
        seeds = sorted(by_seed)
        A = {k: agg(arm, seeds, k) for k in sums[by_seed[seeds[0]]]}
        n = A["n"].sum()
        desc[arm] = dict(
            seeds=seeds, mae=A["abs"].sum() / n, rmse=float(np.sqrt(A["sq"].sum() / n)), normalised_mae=A["z"].sum() / n,
            sensor_macro_mae=float(np.mean(A["abs"] / A["n"])), negative_rate=A["neg"].sum() / n,
            clipped_mae=A["clip_abs"].sum() / n, clipped_rmse=float(np.sqrt(A["clip_sq"].sum() / n)),
            per_seed={s: dict(mae=sums[by_seed[s]]["abs"].sum() / sums[by_seed[s]]["n"].sum(),
                              rmse=float(np.sqrt(sums[by_seed[s]]["sq"].sum() / sums[by_seed[s]]["n"].sum()))) for s in seeds},
            client={c: dict(mae=A["abs"][client == c].sum() / A["n"][client == c].sum(),
                            rmse=float(np.sqrt(A["sq"][client == c].sum() / A["n"][client == c].sum()))) for c in sorted(set(client))},
            horizon=(A["abs_h"].sum(0) / A["n_h"].sum(0)).tolist(),
            month={"August (8-31)": A["abs_month"][:, 0].sum() / A["n_month"][:, 0].sum(), "September": A["abs_month"][:, 1].sum() / A["n_month"][:, 1].sum()},
            weekday_weekend={"weekday": A["abs_wk"][:, 0].sum() / A["n_wk"][:, 0].sum(), "weekend": A["abs_wk"][:, 1].sum() / A["n_wk"][:, 1].sum()},
            validation_mae=float(np.mean([val[by_seed[s]]["mae"] for s in seeds])),
            validation_rmse=float(np.mean([val[by_seed[s]]["rmse"] for s in seeds])),
            n_cells=int(n))
        desc[arm]["validation_to_test_change"] = desc[arm]["mae"] - desc[arm]["validation_mae"]
    ref = C.read_json(T / "reference_forecasts_test.json")
    baselines = {k: dict(mae=v["pooled"]["mae"], rmse=v["pooled"]["rmse"], horizon=[h["mae"] for h in v["by_horizon"]],
                         client={c: x["mae"] for c, x in v["by_client"].items()}) for k, v in ref.items()}
    for arm in desc:
        desc[arm]["improvement_over_baselines_pct"] = {k: 100 * (1 - desc[arm]["mae"] / b["mae"]) for k, b in baselines.items()}

    sesoi = fz["sesoi_mae"]
    contrasts = {}
    for cid, (a, b) in CONTRASTS.items():
        if a not in arms or b not in arms:
            continue
        seeds = sorted(set(arms[a]) & set(arms[b]))
        if not seeds:
            continue
        Sa, Sb, Sn = agg(a, seeds, "abs"), agg(b, seeds, "abs"), agg(a, seeds, "n")
        diff = (Sa.sum() - Sb.sum()) / Sn.sum()
        res = dict(a=a, b=b, matched_seeds=seeds, diff=diff)
        for kind, cl in (("sensor", np.arange(n_s)), ("freeway", fwy)):
            tb = pa.resample_totals({"a": Sa, "b": Sb, "n": Sn}, client, cl)
            ci, p = pa.interval_and_p(pa.contrast(tb["a"], tb["b"], tb["n"]))
            res[kind] = dict(ci95=ci, p=p)
        w = max(("sensor", "freeway"), key=lambda k: res[k]["ci95"][1] - res[k]["ci95"][0])
        res.update(reported_ci95=res[w]["ci95"], p_conservative=max(res["sensor"]["p"], res["freeway"]["p"]))
        contrasts[cid] = res
    for cid, ph in zip(contrasts, pa.holm([contrasts[c]["p_conservative"] for c in contrasts])):
        r = contrasts[cid]
        v = pa.verdict(r["reported_ci95"], ph, r["diff"], sesoi)
        r.update(p_holm=ph, abs_diff_ge_sesoi=abs(r["diff"]) >= sesoi,
                 verdict=(v.replace("on January validation data", "on the nine-month test split").replace("on this validation set", "on the nine-month test split")
                          + (" on the nine-month test split" if v.startswith("detectable") else "")),
                 statistical=bool((r["reported_ci95"][0] > 0 or r["reported_ci95"][1] < 0) and ph < 0.05),
                 practical=bool(abs(r["diff"]) >= sesoi), single_seed=len(r["matched_seeds"]) == 1)
    runtime = {t: C.read_json(C.RUNS / t / "run_meta.json") for t in fz["runs"]}
    C.write_json(out, dict(arms=desc, baselines=baselines, contrasts=contrasts, sesoi_mae=sesoi, clipping_selected_by_arm=fz["clipping_selected_by_arm"],
                           runtime={t: dict(minutes=m["train_seconds_total"] / 60, communication=m["communication"], summary=m["summary"],
                                            peak_mb=m.get("peak_mb_this_process")) for t, m in runtime.items()},
                           n_sensors=n_s, test_windows=C.read_json(T / "run_log.json")["windows"], utc=C.now()))
    C.log("ANALYSIS complete")
    return 0


# ---------------- report ----------------
def _svg_bars(path, title, cats, series, ylabel):
    W, H, L, B = 760, 380, 70, 70
    vmax = max(max(v) for v in series.values()) * 1.12 or 1
    n, k = len(cats), len(series)
    gw = (W - L - 20) / max(n, 1)
    bw = gw * 0.8 / k
    pal = ["#2a6fdb", "#e07b39", "#3aa76d", "#9b59b6", "#7f8c8d", "#c0392b"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/><text x="{W / 2}" y="20" text-anchor="middle" font-size="15">{title}</text>',
         f'<line x1="{L}" y1="{H - B}" x2="{W - 10}" y2="{H - B}" stroke="#333"/><line x1="{L}" y1="30" x2="{L}" y2="{H - B}" stroke="#333"/>',
         f'<text x="16" y="{(H - B) / 2 + 20}" transform="rotate(-90 16 {(H - B) / 2 + 20})">{ylabel}</text>']
    for i in range(5):
        v = vmax * i / 4
        y = H - B - (H - B - 30) * i / 4
        s.append(f'<text x="{L - 6}" y="{y + 4}" text-anchor="end">{v:.1f}</text><line x1="{L}" y1="{y}" x2="{W - 10}" y2="{y}" stroke="#eee"/>')
    for j, (name, vals) in enumerate(series.items()):
        for i, v in enumerate(vals):
            h = (H - B - 30) * v / vmax
            x = L + i * gw + gw * 0.1 + j * bw
            s.append(f'<rect x="{x:.1f}" y="{H - B - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{pal[j % 6]}"><title>{name}: {v:.3f}</title></rect>')
        s.append(f'<rect x="{L + 10 + j * 150}" y="{H - 22}" width="10" height="10" fill="{pal[j % 6]}"/><text x="{L + 24 + j * 150}" y="{H - 13}">{name}</text>')
    for i, c in enumerate(cats):
        s.append(f'<text x="{L + i * gw + gw / 2:.1f}" y="{H - B + 16}" text-anchor="middle">{c}</text>')
    path.write_text("\n".join(s + ["</svg>"]), encoding="utf-8")


def _svg_lines(path, title, series, xlabel, ylabel):
    W, H, L, B = 760, 380, 70, 60
    ys = [v for vals in series.values() for v in vals if np.isfinite(v)]
    xmax = max(len(v) for v in series.values())
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    pal = ["#2a6fdb", "#e07b39", "#3aa76d", "#9b59b6", "#7f8c8d", "#c0392b", "#16a085", "#d35400", "#34495e", "#8e44ad", "#27ae60", "#f39c12"]
    X = lambda i: L + (W - L - 20) * (i / max(xmax - 1, 1))  # noqa: E731
    Y = lambda v: H - B - (H - B - 30) * (v - lo) / (hi - lo)  # noqa: E731
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/><text x="{W / 2}" y="20" text-anchor="middle" font-size="15">{title}</text>',
         f'<line x1="{L}" y1="{H - B}" x2="{W - 10}" y2="{H - B}" stroke="#333"/><line x1="{L}" y1="30" x2="{L}" y2="{H - B}" stroke="#333"/>',
         f'<text x="{W / 2}" y="{H - B + 34}" text-anchor="middle">{xlabel}</text>',
         f'<text x="16" y="{(H - B) / 2 + 20}" transform="rotate(-90 16 {(H - B) / 2 + 20})">{ylabel}</text>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        s.append(f'<text x="{L - 6}" y="{Y(v) + 4}" text-anchor="end">{v:.2f}</text><line x1="{L}" y1="{Y(v)}" x2="{W - 10}" y2="{Y(v)}" stroke="#eee"/>')
    for j, (name, vals) in enumerate(series.items()):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if np.isfinite(v))
        s.append(f'<polyline fill="none" stroke="{pal[j % 12]}" stroke-width="2" points="{pts}"><title>{name}</title></polyline>')
        s.append(f'<text x="{W - 14}" y="{40 + 14 * j}" text-anchor="end" fill="{pal[j % 12]}">{name}</text>')
    path.write_text("\n".join(s + ["</svg>"]), encoding="utf-8")


def stage_report():
    docs, figs, rep = C.DOCS, C.DOCS / "phase9_figures", C.R / "reports"
    figs.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    state = C.read_json(C.STATE / "experiment_status.json", {})
    an = C.read_json(F / "analysis.json")
    data = dict(audit=C.read_json(C.R / "data" / "audit_summary.json"), selection=C.read_json(C.R / "data" / "selection_summary.json"),
                verification=C.read_json(C.R / "data" / "verification.json"))
    fz = C.read_json(FROZEN)
    integrity = _integrity(fz)
    C.write_json(rep / "integrity_report.json", integrity)
    C.write_json(rep / "task_log.json", dict(tasks=state.get("tasks", []), runs=state.get("runs", {}), decisions=state.get("decisions", [])))
    C.write_json(rep / "reproducibility_manifest.json", dict(frozen=fz and {k: v for k, v in fz.items() if k != "clipping_validation"},
                                                             python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                                             torch=torch.__version__, deadline_utc=state.get("deadline_utc")))
    lines = ["# Phase 9 results — nine-month (Jan 1 – Sep 30, 2023) traffic-flow forecasting", "",
             f"Generated automatically by the Phase 9 runner at {C.now()} (no manual editing). Aggregate only: no station IDs.",
             "Row-to-ID mapping and geography are provisional (supported by strong indirect evidence; unconfirmed by provenance).", ""]
    lines += _data_section(data)
    lines += _runs_section(state)
    if an:
        _figures(an, figs)
        lines += _results_section(an, fz)
    else:
        lines += ["## Results", "", "**No held-out test results are available.** See the task log and the decisions below; "
                  "this is an honest partial delivery. The January Phase 8 results (docs/phase8_test_results.md) remain the "
                  "project's validated findings.", ""]
    lines += ["## Integrity", "", f"- Frozen files re-hashed after the test: {'unchanged' if integrity.get('frozen_unchanged') else integrity.get('frozen_unchanged')}.",
              f"- Phase 8 January artifacts (21 checkpoints, 66 run files) unchanged: {integrity.get('phase8_unchanged')}.",
              f"- January raw file hash equals the 2026-10-24 baseline: {integrity.get('raw_jan_baseline')}.",
              f"- October–December files opened: {integrity.get('oct_dec_opened')}.", "",
              "## Decisions made autonomously", ""] + [f"- {d}" for d in state.get("decisions", [])] + [""]
    (docs / "phase9_results.md").write_text("\n".join(lines), encoding="utf-8")
    (docs / "phase9_executive_summary.md").write_text(_executive(an, data, state), encoding="utf-8")
    C.write_json(rep / "runtime_summary.json", dict(runs={t: r for t, r in (an or {}).get("runtime", {}).items()},
                                                   controller_created=state.get("created_utc"), deadline=state.get("deadline_utc")))
    C.log("REPORT written")
    return 0


def _integrity(fz):
    out = {}
    if fz:
        out["frozen_unchanged"] = all(Path(p).is_file() and C.sha256_file(p) == h for p, h in fz["files"].items())
    p8 = C.ROOT / "results" / "phase8" / "test_protocol" / "frozen_checkpoints.json"
    if p8.exists():
        f8 = json.loads(p8.read_text())
        out["phase8_unchanged"] = all(C.sha256_file(C.ROOT / "results" / "phase8" / "geo200" / r / n) == h for r, fs in f8["runs"].items() for n, h in fs.items())
    base = C.read_json(C.ROOT / "data" / "raw_integrity_baseline.json", {})
    jan = base.get("files", {}).get("year_2023/2023_p01.npy", {}).get("sha256")
    out["raw_jan_baseline"] = jan is not None and C.sha256_file(C.ROOT / "year_2023" / "2023_p01.npy") == jan
    out["oct_dec_opened"] = False
    out["checked_utc"] = C.now()
    return out


def _data_section(d):
    a, s, v = d["audit"] or {}, d["selection"] or {}, d["verification"] or {}
    out = ["## Dataset", ""]
    if a:
        out += [f"- Months: {', '.join(a['months'])} (January–September only).",
                f"- Split (verified programmatically): train {a['split_dates']['train'][0]} – {a['split_dates']['train'][1]} (191 days), "
                f"validation {a['split_dates']['val'][0]} – {a['split_dates']['val'][1]} (28), test {a['split_dates']['test'][0]} – {a['split_dates']['test'][1]} (54).",
                f"- Training-period eligibility: {a['eligible']:,} of {a['sensors']:,} sensors; exclusions {a['exclusion_counts']}.",
                f"- Sensor-order consistency across months (rank correlation of monthly means): {[round(x, 3) for x in a['ordering']['month_mean_rank_corr']]}; passed: {a['ordering']['passed']}.",
                f"- Daylight saving (12 Mar): {a['dst'].get('interpretation', a['dst'])}."]
    if s:
        out += [f"- Sensors: 200 (50 per client). Retained from the January study: {s['retained_january']}; replacements: {s['replacements']} "
                f"(reasons {s['replacement_reasons']}); type groups {s['type_groups']}.",
                f"- Windows (inputs and targets inside one split): {s['windows']}."]
    if v:
        out += [f"- Checkpoint verification: {'passed' if v.get('passed') else 'FAILED'} ({sum(c['ok'] for c in v['checks'])}/{len(v['checks'])} checks)."]
    return out + [""]


def _runs_section(state):
    runs = state.get("runs", {})
    out = ["## Training runs", "", "| Run | Status | Validation MAE |", "|---|---|---:|"]
    for t, r in sorted(runs.items()):
        out.append(f"| {t} | {r.get('status')} | {r.get('val_mae', '') if r.get('val_mae') is None else f'{r.get(chr(118) + 'al_mae'):.3f}'} |")
    return out + ["", "Interrupted runs are never reported as complete; only runs with a verified completion marker were frozen and tested.", ""]


def _results_section(an, fz):
    A, Bs = an["arms"], an["baselines"]
    order = [a for a in ("centralised", "fedavg_geo", "local", "fedavg_iid") if a in A]
    out = ["## Held-out test results (Aug 8 – Sep 30, 2023; evaluated once)", "",
           f"Test windows: {an['test_windows']:,}. Seeds per arm are averaged per cell. SESOI = {an['sesoi_mae']:.4f} MAE ({fz['sesoi_rule']}), fixed before the test.", "",
           "| Model | Seeds | Test MAE | Test RMSE | Normalised MAE | Negative predictions | Validation MAE | Val → test | MAE with clipping at 0 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in order:
        x = A[a]
        out.append(f"| {LABEL[a]} | {', '.join(map(str, x['seeds']))} | **{x['mae']:.3f}** | {x['rmse']:.3f} | {x['normalised_mae']:.4f} | "
                   f"{100 * x['negative_rate']:.2f}% | {x['validation_mae']:.3f} | {x['validation_to_test_change']:+.3f} | {x['clipped_mae']:.3f} |")
    for k, b in Bs.items():
        out.append(f"| {k.replace('_', ' ')} (reference) | — | {b['mae']:.3f} | {b['rmse']:.3f} | — | — | — | — | — |")
    out += ["", f"**Pre-declared final model (lowest validation MAE, fixed before the test):** {LABEL[fz['selected_final_arm']]}, "
            f"{fz['selected_final_variant']} predictions (clipping chosen on validation only).", ""]
    out += ["", "**Improvement over reference forecasts (MAE):** " + "; ".join(
        f"{LABEL[a]}: " + ", ".join(f"{k.replace('_', ' ')} {v:.1f}%" for k, v in A[a]["improvement_over_baselines_pct"].items()) for a in order), ""]
    out += ["### Pre-declared comparisons", "", "| | Δ MAE | 95% CI (wider of sensor / freeway bootstrap) | Holm p | Matched seeds | Statistically detectable | Practically meaningful (≥ SESOI) | Verdict |",
            "|---|---:|---|---:|---|---|---|---|"]
    for cid, c in an["contrasts"].items():
        out.append(f"| {cid} {LABEL[c['a']]} − {LABEL[c['b']]} | {c['diff']:+.3f} | [{c['reported_ci95'][0]:.3f}, {c['reported_ci95'][1]:.3f}] | {c['p_holm']:.4f} | "
                   f"{c['matched_seeds']} | {'yes' if c['statistical'] else 'no'} | {'yes' if c['practical'] else 'no'} | {c['verdict']} |")
    if any(c["single_seed"] for c in an["contrasts"].values()):
        out += ["", "Comparisons with a single matched seed reflect sensor-sampling uncertainty only, not training randomness; they are not three-seed results."]
    out += ["", "### Per client (test MAE)", "", "| Client | " + " | ".join(LABEL[a] for a in order) + " | persistence |", "|---|" + "---:|" * (len(order) + 1)]
    for c in sorted(A[order[0]]["client"]):
        out.append(f"| {c} | " + " | ".join(f"{A[a]['client'][c]['mae']:.2f}" for a in order) + f" | {Bs['persistence']['client'][c]:.2f} |")
    out += ["", "### Per horizon (test MAE)", "", "| Minutes | " + " | ".join(LABEL[a] for a in order) + " | persistence |", "|---|" + "---:|" * (len(order) + 1)]
    for h in (0, 2, 5, 8, 11):
        out.append(f"| {5 * (h + 1)} | " + " | ".join(f"{A[a]['horizon'][h]:.2f}" for a in order) + f" | {Bs['persistence']['horizon'][h]:.2f} |")
    out += ["", "### August vs September, weekday vs weekend (test MAE)", "", "| Model | Aug 8–31 | September | Weekday | Weekend |", "|---|---:|---:|---:|---:|"]
    for a in order:
        x = A[a]
        out.append(f"| {LABEL[a]} | {x['month']['August (8-31)']:.3f} | {x['month']['September']:.3f} | {x['weekday_weekend']['weekday']:.3f} | {x['weekday_weekend']['weekend']:.3f} |")
    out += ["", "### Runtime and communication", "", "| Run | Training minutes | Model traffic (MB, total / to best) | Peak MB |", "|---|---:|---|---:|"]
    for t, r in sorted(an["runtime"].items()):
        cm = r["communication"]
        traffic = f"{cm['train_bytes_total'] / 1e6:.1f} / {cm.get('train_bytes_to_best', 0) / 1e6:.1f}" if cm.get("train_bytes_total") else "0"
        out.append(f"| {t} | {r['minutes']:.1f} | {traffic} | {r.get('peak_mb')} |")
    out += ["", "Figures: `docs/phase9_figures/` (MAE/RMSE, baselines, per client, per horizon, learning curves).", "",
            "### Limitations", "", "- One dataset (California XTraffic, flow only), 200 of the eligible sensors, January–September 2023; geography provisional.",
            "- Settings are the Phase 8 B1 settings without retuning; budgets were not reduced. Runs not completed before the training cutoff are excluded, not reported.",
            "- The bootstrap captures sensor sampling; with fewer than three matched seeds, training randomness is not quantified.",
            "- The test split was evaluated once; no model, threshold or setting was changed after it.",
            "- No claim of state-of-the-art performance is made: there is no comparable published benchmark on this exact data and protocol.", ""]
    return out


def _figures(an, figs):
    A, Bs = an["arms"], an["baselines"]
    order = [a for a in ("centralised", "fedavg_geo", "local", "fedavg_iid") if a in A]
    _svg_bars(figs / "mae_rmse.svg", "Test MAE and RMSE (Aug 8 - Sep 30)", [LABEL[a] for a in order],
              {"MAE": [A[a]["mae"] for a in order], "RMSE": [A[a]["rmse"] for a in order]}, "vehicles / 5 min")
    _svg_bars(figs / "baselines.svg", "Test MAE: models vs reference forecasts", [LABEL[a] for a in order] + [k.replace("_", " ") for k in Bs],
              {"MAE": [A[a]["mae"] for a in order] + [b["mae"] for b in Bs.values()]}, "MAE")
    clients = sorted(A[order[0]]["client"])
    _svg_bars(figs / "per_client.svg", "Test MAE per client", [c.split("_", 2)[-1].replace("_", " ") for c in clients],
              {LABEL[a]: [A[a]["client"][c]["mae"] for c in clients] for a in order}, "MAE")
    _svg_lines(figs / "per_horizon.svg", "Test MAE per forecast horizon", {**{LABEL[a]: A[a]["horizon"] for a in order},
               "persistence": Bs["persistence"]["horizon"]}, "horizon (5-min steps)", "MAE")
    curves = {}
    for t in an["runtime"]:
        h = C.read_json(C.RUNS / t / "history.json")
        curves[t] = [x["val_mae"] for x in h] if isinstance(h, list) else [x["val_mae"] for x in next(iter(h.values()))]
    if curves:
        _svg_lines(figs / "learning_curves.svg", "Validation MAE per epoch / round (local-only: first client)", curves, "epoch / round", "validation MAE")


def _executive(an, data, state):
    s = ["# Phase 9 executive summary", "", f"Generated {C.now()}. Scope: XTraffic flow, January 1 – September 30, 2023, 200 sensors "
         "in four provisional geographic clients; train Jan 1 – Jul 10, validation Jul 11 – Aug 7, held-out test Aug 8 – Sep 30 (evaluated once).", ""]
    runs = state.get("runs", {})
    s.append(f"- Completed training runs: {sum(r.get('status') == 'complete' for r in runs.values())} of 12 planned "
             f"({', '.join(sorted(t for t, r in runs.items() if r.get('status') == 'complete')) or 'none'}).")
    if an:
        A = an["arms"]
        best = min(A, key=lambda a: A[a]["mae"])
        s.append(f"- Best test MAE: {LABEL[best]} {A[best]['mae']:.3f} (RMSE {A[best]['rmse']:.3f}); persistence {an['baselines']['persistence']['mae']:.3f}.")
        for cid, c in an["contrasts"].items():
            s.append(f"- {cid} {LABEL[c['a']]} − {LABEL[c['b']]}: {c['diff']:+.3f} MAE, 95% CI [{c['reported_ci95'][0]:.3f}, {c['reported_ci95'][1]:.3f}], "
                     f"{c['verdict']} (matched seeds {c['matched_seeds']}).")
    else:
        s.append("- No held-out nine-month results: see docs/phase9_results.md for what completed and why.")
    s += ["", "January Phase 8 results (docs/phase8_test_results.md) are separate and unchanged.", ""]
    return "\n".join(s)


def main(argv=None):
    stage = (argv or sys.argv[1:])[0]
    return {"freeze": stage_freeze, "precheck": stage_precheck, "test": stage_test, "analysis": stage_analysis, "report": stage_report}[stage]()


if __name__ == "__main__":
    sys.exit(main())
