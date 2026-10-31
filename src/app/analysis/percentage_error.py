"""Read-only percentage-error analysis of the frozen Phase 9 held-out test predictions.

    .venv\\Scripts\\python.exe app/analysis/percentage_error.py

Reads results/phase9/final/test/<run>/predictions_test.npz (written once by the Phase 9 test stage) and the frozen
reference-forecast scores; runs no model, changes no file there, and writes only the aggregate app/data/percentage_error.json.

Definitions (all over the same observed target cells: mask = finite ground truth):
  WAPE  = 100 * sum|actual - predicted| / sum|actual|
  sMAPE = 100 * mean( 2|actual - predicted| / (|actual| + |predicted|) ), cells where actual = predicted = 0 excluded
Arm values average the three seeds per cell (as in the Phase 9 analysis); per-seed values are reported too.
MAPE is not reported: it is undefined at zero flow and explodes at low flow (share of such cells is reported instead).
Persistence WAPE uses its frozen pooled/per-client/per-horizon MAE times the same cell counts (identical masks);
its sMAPE needs per-cell predictions, which were not saved, so it is reported as unavailable.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "results" / "phase9" / "final" / "test"
OUT = ROOT / "app" / "data" / "percentage_error.json"
ARMS = {"local": "Local-only GRU", "centralised": "Centralised GRU", "fedavg_iid": "IID FedAvg", "fedavg_geo": "Geographic FedAvg"}
CLIENTS = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"]


def run_sums(tag, clip):
    with np.load(TEST / tag / "predictions_test.npz") as z:
        p, a, m, client = z["y_hat"].astype(np.float64), z["y_true"].astype(np.float64), z["mask"], z["client"]
        names = list(z["client_names"])
    if clip:
        p = np.maximum(p, 0.0)
    a = np.where(m, a, 0.0)
    err = np.where(m, np.abs(a - p), 0.0)
    den = np.abs(a) + np.abs(p)
    sm_ok = m & (den > 0)
    sm = np.where(sm_ok, 2 * err / np.where(den > 0, den, 1), 0.0)
    cl = np.array([names[c] for c in range(len(names))])[client]
    out = dict(err=err.sum(), act=np.abs(a).sum(), n=int(m.sum()), sm=sm.sum(), sm_n=int(sm_ok.sum()),
               err_h=err.sum(0), act_h=np.abs(a).sum(0), sm_h=sm.sum(0), sm_n_h=sm_ok.sum(0),
               zero_cells=int((m & (a == 0)).sum()), low_cells=int((m & (a < 5)).sum()))
    for c in CLIENTS:
        k = cl == c
        out[c] = dict(err=err[k].sum(), act=np.abs(a[k]).sum(), sm=sm[k].sum(), sm_n=int(sm_ok[k].sum()))
    return out


def summarise(sums):
    """Arm-level metrics from per-seed sums (seed-averaged per cell == averaged numerators on identical cells)."""
    err, act = np.mean([s["err"] for s in sums]), sums[0]["act"]
    res = dict(wape_pct=100 * err / act, smape_pct=float(np.mean([100 * s["sm"] / s["sm_n"] for s in sums])),
               mae=err / sums[0]["n"],
               per_seed_wape_pct=[100 * s["err"] / s["act"] for s in sums],
               by_horizon_wape_pct=(100 * np.mean([s["err_h"] for s in sums], 0) / sums[0]["act_h"]).tolist(),
               by_horizon_smape_pct=np.mean([100 * s["sm_h"] / s["sm_n_h"] for s in sums], 0).tolist(),
               by_client_wape_pct={c: 100 * np.mean([s[c]["err"] for s in sums]) / sums[0][c]["act"] for c in CLIENTS},
               by_client_smape_pct={c: float(np.mean([100 * s[c]["sm"] / s[c]["sm_n"] for s in sums])) for c in CLIENTS})
    return res


def main():
    out = dict(definitions="Definitions" + __doc__.split("Definitions", 1)[1].rstrip(), source="frozen Phase 9 held-out test predictions (Aug 8 - Sep 30, 2023)", arms={})
    ref_sums = None
    for arm, label in ARMS.items():
        raw = [run_sums(f"{arm}_seed{s}", clip=False) for s in (0, 1, 2)]
        ref_sums = ref_sums or raw[0]
        out["arms"][arm] = dict(label=label, raw=summarise(raw))
        if arm == "local":
            clipped = [run_sums(f"{arm}_seed{s}", clip=True) for s in (0, 1, 2)]
            out["arms"][arm]["clipped"] = summarise(clipped)
            out["deployed_local_seed1_clipped"] = dict(wape_pct=100 * clipped[1]["err"] / clipped[1]["act"],
                                                       smape_pct=100 * clipped[1]["sm"] / clipped[1]["sm_n"],
                                                       mae=clipped[1]["err"] / clipped[1]["n"])
            print(f"{label} clipped: WAPE {out['arms'][arm]['clipped']['wape_pct']:.3f}%  sMAPE {out['arms'][arm]['clipped']['smape_pct']:.3f}%")
        print(f"{label}: WAPE {out['arms'][arm]['raw']['wape_pct']:.3f}%  sMAPE {out['arms'][arm]['raw']['smape_pct']:.3f}%  MAE check {out['arms'][arm]['raw']['mae']:.3f}")
    s = ref_sums
    ref = json.loads((TEST / "reference_forecasts_test.json").read_text())
    for name, r in ref.items():
        if r["pooled"]["n"] != s["n"]:
            raise SystemExit(f"{name}: cell count {r['pooled']['n']} differs from model cells {s['n']}")
        out.setdefault("baselines", {})[name] = dict(
            wape_pct=100 * r["pooled"]["mae"] * s["n"] / s["act"], smape_pct=None, mae=r["pooled"]["mae"],
            by_horizon_wape_pct=[100 * h["mae"] * h["n"] / s["act_h"][i] for i, h in enumerate(r["by_horizon"])],
            by_client_wape_pct={c: 100 * r["by_client"][c]["mae"] * r["by_client"][c]["n"] / s[c]["act"] for c in CLIENTS})
    out["cells"] = dict(observed=s["n"], actual_zero=s["zero_cells"], actual_below_5=s["low_cells"],
                        mean_actual_flow=s["act"] / s["n"])
    final = out["arms"]["local"]["clipped"]["wape_pct"]
    out["target"] = dict(wape_target_pct=6.0, achieved=bool(final <= 6.0), final_model_wape_pct=final, gap_pct_points=final - 6.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"persistence WAPE {out['baselines']['persistence']['wape_pct']:.3f}%; mean actual flow {out['cells']['mean_actual_flow']:.2f}; "
          f"zero cells {s['zero_cells'] / s['n']:.2%}; target 6% achieved: {out['target']['achieved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
