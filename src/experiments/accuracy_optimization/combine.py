"""Phases F and G: ensembles and residual correction, fitted and scored with 2-fold blocked cross-validation inside
the validation period (fold A = Jul 11-24, fold B = Jul 25 - Aug 7). Every number reported is out-of-fold: weights or
correction models fitted on one fold are scored on the other, then the folds swap. Test data is never read.

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/combine.py EXP_ID [EXP_ID ...] [--residual]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aopt import data as D  # noqa: E402
from aopt.metrics import full_report  # noqa: E402

OUT = HERE.parents[1] / "results" / "accuracy_optimization"
SPLIT_T = 55008 + 14 * 288


def load_truth():
    with np.load(D.DATA / "window_index.npz") as w:
        v = w["val"].astype(np.int64)
    with np.load(D.DATA / "pilot_preprocessed.npz") as d:
        flow = d["flow_raw"][:, :63072].astype(np.float64)
    y = flow[v[:, 0][:, None], v[:, 1][:, None] + np.arange(12)]
    return v, flow, y


def l1_weights(P, a, m, iters=400):
    """Convex weights minimising total absolute error (exponentiated subgradient on the simplex)."""
    k = P.shape[0]
    w = np.full(k, 1 / k)
    Pm = P[:, m]
    am = a[m]
    for i in range(iters):
        r = np.tensordot(w, Pm, 1) - am
        g = (Pm * np.sign(r)).mean(1) / max(am.mean(), 1)
        w = w * np.exp(-2.0 / np.sqrt(i + 1) * g)
        w /= w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    v, flow, y = load_truth()
    m = np.isfinite(y)
    a = np.where(m, y, 0.0)
    P = np.stack([np.load(OUT / "validation_predictions" / f"{i}.npy").astype(np.float64) for i in args.ids])
    foldA = v[:, 1] + 11 < SPLIT_T
    # windows straddling the fold boundary are scored with the fold whose weights do not use them
    folds = [(foldA, ~foldA), (~foldA, foldA)]
    pred = np.zeros_like(a)
    report = {"members": args.ids, "folds": "blocked: A = targets ending before 2023-07-25, B = the rest"}
    weights = []
    for fit, score in folds:
        w = l1_weights(P[:, fit], a[fit], m[fit]) if len(args.ids) > 1 else np.ones(1)
        weights.append(w.tolist())
        pred[score] = np.tensordot(w, P[:, score], 1)
    report["weights_per_fold"] = weights
    report["mean_member_wape_pct"] = [100 * float(np.abs(np.where(m, p - a, 0)).sum() / a.sum()) for p in P]
    if args.residual:
        pred = residual_correct(pred, v, flow, a, m, folds, report)
    import pandas as pd
    sel = pd.read_csv(D.DATA / "sensor_selection.csv", usecols=["proposed_client"])
    client = np.array([D.CLIENTS.index(c) for c in sel["proposed_client"]])[v[:, 0]]
    rep = full_report(np.maximum(pred, 0), a, m, v[:, 0], v[:, 1], client, D.CLIENTS)
    rep.pop("by_hour_share_of_abs_error", None)
    report["validation_oof"] = rep
    name = args.name or ("ens_" + "+".join(args.ids) + ("_res" if args.residual else ""))
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{name}.json").write_text(json.dumps(report, indent=1, default=float))
    with open(OUT / "registry.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(exp_id=name, algorithm="ensemble" + ("+residual" if args.residual else ""), architecture="combination",
                                 status="completed", members=args.ids, validation=rep, weights_per_fold=weights,
                                 config=dict(evaluation="2-fold blocked out-of-fold within validation")), default=float) + "\n")
    print(f"{name}: OOF validation WAPE {rep['wape_pct']:.3f}%  MAE {rep['mae']:.3f}  members {[round(x, 3) for x in report['mean_member_wape_pct']]}  weights {weights}")


def residual_correct(pred, v, flow, a, m, folds, report):
    """LightGBM on base-forecast residuals with features known at forecast time; out-of-fold within validation."""
    import lightgbm as lgb
    N = len(v)
    t0 = v[:, 1]
    last = np.nan_to_num(flow[v[:, 0], t0 - 1])
    mean12 = np.nanmean(flow[v[:, 0][:, None], t0[:, None] + np.arange(-12, 0)], 1)
    trend = np.nan_to_num(flow[v[:, 0], t0 - 1] - flow[v[:, 0], t0 - 4])
    rows = []
    for h in range(12):
        t = t0 + h
        rows.append(np.column_stack([pred[:, h], np.full(N, h), np.nan_to_num(last), np.nan_to_num(mean12), trend,
                                     (t % 288), (6 + t // 288) % 7, v[:, 0]]))
    X = np.stack(rows, 1)                      # (N, 12, F)
    out = pred.copy()
    rng = np.random.default_rng(0)
    gains = []
    for fit, score in folds:
        Xf, rf, mf = X[fit].reshape(-1, X.shape[2]), (a[fit] - pred[fit]).ravel(), m[fit].ravel()
        keep = np.nonzero(mf)[0]
        keep = rng.choice(keep, min(3_000_000, len(keep)), replace=False)
        model = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
        model.fit(Xf[keep], rf[keep], categorical_feature=[7])
        Xs = X[score].reshape(-1, X.shape[2])
        out[score] = pred[score] + model.predict(Xs).reshape(-1, 12)
        gains.append(model.feature_importances_.tolist())
    report["residual_model"] = dict(kind="LightGBM l1, 300 trees", features=["base", "horizon", "last", "mean12", "trend3", "slot", "weekday", "sensor"],
                                    importances=gains)
    return out


if __name__ == "__main__":
    main()
