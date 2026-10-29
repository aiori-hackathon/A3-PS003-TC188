"""ONE-TIME independent evaluation on October - December 2023 (docs/accuracy_optimization_protocol.md).

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/eval_untouched.py CANDIDATE_EXP_ID [--seed-model KEY]

Runs once, after selection is final. It refuses to run again: a marker file records the first run.
Scores (a) the selected candidate, (b) the frozen Phase 9 deployed model (local_seed1, clipped) and (c) persistence
on every window whose 12 targets lie in Oct 1 - Dec 31, 2023 (at least one observed target).

Preprocessing is the study's own: the frozen training scaler and time-of-day medians, the training weekday x slot
profile, and causal filling (forward fill <= 30 min within the Oct - Dec segment, then the training time-of-day median,
then the training mean). January - September values come from the frozen nine-month checkpoint unchanged; only the
selected sensors' flow rows of 2023_p10 ... p12 are read (memory-mapped; the raw files are never modified).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from aopt import data as D  # noqa: E402
from aopt.metrics import full_report  # noqa: E402
from aopt.train import Config, make_model  # noqa: E402
from aiori import baselines, config as acfg, preprocessing  # noqa: E402

OUT = ROOT / "results" / "accuracy_optimization" / "untouched_oct_dec"
MARKER = OUT / "OPENED.json"
DAYS = {10: 31, 11: 30, 12: 31}
SPD = 288


def build(device):
    base = D.load(device, min_t0=0)               # frozen Jan - Aug 7 arrays, scaler, profile (training-only)
    with np.load(D.DATA / "pilot_preprocessed.npz") as d:
        flow9 = d["flow_raw"].astype(np.float64)    # Jan - Sep (frozen checkpoint)
        z9, imp9, tod = d["inputs_scaled"], d["input_imputed"], d["tod_median"]
        rows = d["array_index"]
    sc = json.loads((D.DATA / "scaler_parameters.json").read_text())
    mean, std = np.asarray(sc["mean"]), np.asarray(sc["std"])
    order = np.argsort(rows)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(rows))
    parts = []
    for m in (10, 11, 12):
        a = np.load(ROOT / "year_2023" / f"2023_p{m:02d}.npy", mmap_mode="r")
        assert a.shape == (16972, DAYS[m] * SPD, 3) and a.dtype == np.float32, a.shape
        parts.append(np.asarray(a[rows[order], :, acfg.FLOW], dtype=np.float64)[inv])
    q4 = np.concatenate(parts, 1)                  # (200, 92 days)
    T9 = flow9.shape[1]
    # Causal fill of the Oct - Dec segment alone, with training statistics (as the study fills each split).
    seg = {"q4": (0, q4.shape[1])}
    train_mean = np.nanmean(flow9[:, :55008], 1)
    filled = q4.copy()
    obs = np.isfinite(q4)
    last = np.full(len(q4), np.nan)
    age = np.zeros(len(q4), np.int64)
    for t in range(q4.shape[1]):
        last = np.where(obs[:, t], q4[:, t], last)
        age = np.where(obs[:, t], 0, age + 1)
        use = ~obs[:, t] & (age <= 6) & np.isfinite(last)
        filled[use, t] = last[use]
    slots = (T9 + np.arange(q4.shape[1])) % SPD
    still = np.isnan(filled)
    filled[still] = tod[:, slots][still]
    still = np.isnan(filled)
    filled[still] = np.broadcast_to(train_mean[:, None], filled.shape)[still]
    del seg
    z_q4 = (filled - mean[:, None]) / std[:, None]
    flow = np.concatenate([flow9, q4], 1)
    z = np.concatenate([z9, z_q4], 1)
    imp = np.concatenate([imp9, ~obs], 1)
    # Windows: all 12 targets in Oct 1 - Dec 31, at least one observed.
    t0 = np.arange(T9, flow.shape[1] - 11)
    cnt = np.stack([np.isfinite(flow[:, t0 + h]) for h in range(12)], 0).sum(0)
    s, k = np.nonzero(cnt >= 1)
    idx = np.stack([s, t0[k]], 1).astype(np.int64)
    T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)  # noqa: E731
    store = D.Store(z=T(z), obs=T(~imp), flow=T(flow), prof=base.prof, mean=base.mean, std=base.std, client=base.client,
                    train=base.train[:0], val=T(idx, torch.long), val_cells=0.0, device=device, n_sensors=base.n_sensors)
    coverage = float(np.isfinite(q4).mean())
    return store, idx, flow, tod, coverage


@torch.no_grad()
def predict(models, store, f, idx):
    from aopt.train import predict_raw
    return predict_raw(models, store, f, idx).double().cpu().numpy()


def load_candidate(exp_id, store, device):
    ck = ROOT / "results" / "accuracy_optimization" / "checkpoints" / exp_id
    cfg = Config(**json.loads((ck / "config.json").read_text()))
    states = torch.load(ck / "models.pt", map_location=device, weights_only=True)
    models = {}
    for k, sd in states.items():
        m = make_model(cfg, store)
        m.load_state_dict(sd)
        models[None if k == "None" else int(k)] = m
    return models, cfg


def load_frozen(store, device):
    cfg = Config(exp_id="frozen_local_seed1", algorithm="local")
    models = {}
    for c, name in enumerate(D.CLIENTS):
        sd = torch.load(ROOT / "results" / "phase9" / "runs" / "local_seed1" / f"best_model_{name}.pt", map_location=device, weights_only=True)
        m = make_model(cfg, store)
        m.load_state_dict({("body." + k[4:]) if k.startswith("gru.") else ("head.net." + k[5:]): v for k, v in sd.items()})
        models[c] = m
    return models, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    args = ap.parse_args()
    if MARKER.exists():
        sys.exit(f"Refusing: the October-December evaluation already ran ({MARKER.read_text()}).")
    OUT.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps(dict(opened=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), candidate=args.candidate)))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store, idx, flow, tod, coverage = build(dev)
    y = flow[idx[:, 0][:, None], idx[:, 1][:, None] + np.arange(12)]
    m = np.isfinite(y)
    a = np.where(m, y, 0.0)
    client = store.client.cpu().numpy()[idx[:, 0]]
    res = dict(protocol="docs/accuracy_optimization_protocol.md", candidate=args.candidate, n_windows=int(len(idx)),
               observed_cells=int(m.sum()), raw_coverage_oct_dec=coverage)
    preds = {}
    cand, ccfg = load_candidate(args.candidate, store, dev)
    preds["candidate"] = predict(cand, store, ccfg.features(), store.val)
    resid_path = ROOT / "results" / "accuracy_optimization" / "candidate_v1" / "residual_lgbm.txt"
    if resid_path.exists():                       # deployable correction, fitted on validation only
        import lightgbm as lgb
        from aopt import residual as R
        booster = lgb.Booster(model_file=str(resid_path))
        X = R.features(preds["candidate"], idx, flow)
        preds["candidate_plus_residual"] = np.maximum(preds["candidate"] + booster.predict(X.reshape(-1, X.shape[-1])).reshape(-1, 12), 0)
    froz, fcfg = load_frozen(store, dev)
    preds["frozen_local_seed1_clipped"] = predict(froz, store, fcfg.features(), store.val)
    preds["persistence"] = np.maximum(baselines.persistence(flow, tod, idx), 0)
    month = np.where(idx[:, 1] < 78624 + 31 * SPD, "October", np.where(idx[:, 1] < 78624 + 61 * SPD, "November", "December"))
    for k, p in preds.items():
        r = full_report(p, a, m, idx[:, 0], idx[:, 1], client, D.CLIENTS)
        r.pop("by_hour_share_of_abs_error", None)
        r["by_month_wape_pct"] = {mo: 100 * float(np.abs(np.where(m, p - a, 0))[month == mo].sum() / a[month == mo].sum())
                                  for mo in ("October", "November", "December")}
        res[k] = r
        print(f"{k}: Oct-Dec WAPE {r['wape_pct']:.3f}%  MAE {r['mae']:.3f}  RMSE {r['rmse']:.3f}  sMAPE {r['smape_pct']:.3f}%  months {({k2: round(v, 2) for k2, v in r['by_month_wape_pct'].items()})}")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
