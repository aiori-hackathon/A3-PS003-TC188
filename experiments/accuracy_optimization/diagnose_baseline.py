"""Phase A: reproduce the local-only validation baseline and diagnose its errors (validation split only).

    .venv\\Scripts\\python.exe experiments/accuracy_optimization/diagnose_baseline.py

Reads the frozen Phase 9 validation predictions of the deployed final model (local_seed1) and the nine-month
checkpoint arrays; writes only results/accuracy_optimization/reports/baseline_diagnostics.json (git-ignored).
The held-out test predictions and October-December data are never read.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aopt.metrics import full_report, wape  # noqa: E402

RUN = ROOT / "results" / "phase9" / "runs" / "local_seed1" / "predictions_val.npz"
DATA = ROOT / "data" / "processed" / "nine_month" / "pilot_preprocessed.npz"
OUT = ROOT / "results" / "accuracy_optimization" / "reports" / "baseline_diagnostics.json"
SPD = 288


def main():
    with np.load(RUN) as z:
        pred = {k: z[k] for k in ("sensor_pos", "client", "t0", "y_hat", "y_true", "mask")}
        client_names = list(z["client_names"])
    with np.load(DATA) as d:
        flow = d["flow_raw"].astype(np.float64)
        imputed = d["input_imputed"]
    sp, t0 = pred["sensor_pos"].astype(np.int64), pred["t0"].astype(np.int64)
    y, m = pred["y_true"].astype(np.float64), pred["mask"]
    # Re-derive the truth from the raw array: alignment and masks must match exactly.
    cols = t0[:, None] + np.arange(12)
    truth = flow[sp[:, None], cols]
    assert np.array_equal(np.isfinite(truth), m), "mask mismatch"
    assert np.allclose(np.where(m, truth, 0), np.where(m, y, 0)), "truth mismatch"
    assert t0.min() >= 55008 and (t0 + 12).max() <= 63072, "windows leave the validation split"

    out = {"source": "Phase 9 local_seed1 validation predictions (frozen); validation split 2023-07-11..08-07"}
    for clip in (False, True):
        yh = pred["y_hat"].astype(np.float64)
        if clip:
            yh = np.maximum(yh, 0)
        out["clipped" if clip else "raw"] = full_report(yh, truth, m, sp, t0, pred["client"], client_names)
    yh = np.maximum(pred["y_hat"].astype(np.float64), 0)
    a = np.where(m, truth, 0.0)
    e = np.where(m, yh - a, 0.0)

    # Error sources ------------------------------------------------------------
    diag = {}
    # input missingness: share of the 12 input steps that were filled
    miss = imputed[sp[:, None], t0[:, None] + np.arange(-12, 0)].mean(1)
    bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]
    diag["by_input_missing_share"] = {f"{lo:.2f}-{hi:.2f}": dict(wape=wape(e[(miss > lo) & (miss <= hi)], a[(miss > lo) & (miss <= hi)]),
                                                                  share_windows=float(((miss > lo) & (miss <= hi)).mean())) for lo, hi in bins}
    # volume quantiles of the true cell value
    qs = np.quantile(a[m], [0.25, 0.5, 0.75, 0.9])
    edges = [-1, *qs, np.inf]
    vq = {}
    for i in range(len(edges) - 1):
        k = m & (a > edges[i]) & (a <= edges[i + 1])
        vq[f"q{i}"] = dict(range=[float(max(edges[i], 0)), float(edges[i + 1])], wape=float(np.abs(e[k]).sum() / a[k].sum()),
                           share_abs_error=float(np.abs(e[k]).sum() / np.abs(e).sum()), bias=float(e[k].mean()))
    diag["by_volume_quantile"] = vq
    # bias
    diag["bias"] = dict(mean_error=float(e[m].mean()), over_share=float((e[m] > 0).mean()), under_share=float((e[m] < 0).mean()),
                        abs_error_from_over=float(e[e > 0].sum() / np.abs(e).sum()))
    # sensors: concentration of error
    per_s = np.bincount(sp, np.abs(e).sum(1), minlength=200)
    act_s = np.bincount(sp, a.sum(1), minlength=200)
    ws = per_s / np.maximum(act_s, 1)
    order = np.argsort(-per_s)
    diag["sensor_error_concentration"] = dict(top10_share=float(per_s[order[:10]].sum() / per_s.sum()),
                                              top20_share=float(per_s[order[:20]].sum() / per_s.sum()),
                                              sensor_wape_pct_quantiles=dict(zip(["p10", "p50", "p90", "max"],
                                                                                 (100 * np.quantile(ws[act_s > 0], [0.1, 0.5, 0.9, 1.0])).tolist())))
    # Noise floor ---------------------------------------------------------------
    # (1) Poisson count noise: even the true mean rate lambda leaves E|y - lambda| ~ sqrt(2 lambda / pi).
    poisson = float(np.sqrt(2 * a[m] / np.pi).sum() / a[m].sum())
    # (2) Non-causal oracle: the mean of the 4 nearest neighbours (t-2, t-1, t+1, t+2) of every cell, i.e. a
    #     forecaster that knows the future around each step. Optimistic lower reference, not achievable.
    T = flow.shape[1]
    nb = np.stack([np.roll(flow, s, axis=1) for s in (-2, -1, 1, 2)])
    nbm = np.nanmean(nb, axis=0)
    oracle = nbm[sp[:, None], cols]
    ok = m & np.isfinite(oracle)
    diag["noise_floor"] = dict(
        poisson_wape_pct=100 * poisson,
        neighbour_oracle_wape_pct=100 * float(np.abs(oracle[ok] - a[ok]).sum() / a[ok].sum()),
        note="Lower references for the best achievable 5-minute WAPE: Poisson count noise around a perfect rate, and a "
             "non-causal neighbour average that sees t+1 and t+2. Neither is a forecaster.")
    diag["persistence_val_wape_pct"] = 100 * float(np.abs(np.where(m, flow[sp, t0 - 1][:, None] - a, 0)).sum() / a.sum()) \
        if np.isfinite(flow[sp, t0 - 1]).all() else None
    out["diagnostics"] = diag
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    c = out["clipped"]
    print(f"baseline (local_seed1, clipped) validation: WAPE {c['wape_pct']:.3f}%  MAE {c['mae']:.3f}  RMSE {c['rmse']:.3f}  sMAPE {c['smape_pct']:.3f}%")
    print("horizon WAPE:", [round(v, 2) for v in c["by_horizon_wape_pct"]])
    print("client WAPE:", {k: round(v["wape_pct"], 2) for k, v in c["by_client"].items()})
    print("peak/offpeak/weekend:", {k: round(v["wape_pct"], 2) for k, v in c["by_period"].items()})
    print(json.dumps(diag, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
