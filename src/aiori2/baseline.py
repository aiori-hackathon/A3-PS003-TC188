"""Persistence baseline on the validation population the runner uses (validation only; test is never read).

    python -m aiori2.baseline [--history A|B] [--max-windows N|all] [--out FILE]

Forecast = the last input value (causally filled flow at t0-1) repeated for all 12 steps. The inverse-scaled value is
the raw filled flow whatever the scaler, so one number serves every arm. Scored with the runner's Evaluator: same
fixed window subset (seed 0) and the same raw-flow MAE/RMSE/WAPE over valid targets.
"""

import argparse
import json

import torch

from aiori2 import core
from aiori2.run import Evaluator, load_config


def persistence(X, s=None):
    return X[:, -1:, 0].expand(-1, 12)


def evaluate(history, max_windows, device="cpu", split="val"):
    cfg = load_config(None, dict(arm="centralised", history=history))
    ev = Evaluator(core.load_config()["_out"], cfg, max_windows, split)
    r = ev.evaluate(persistence, torch.device(device))
    return dict(baseline="persistence (last filled input value)", history=history, windows=r["windows"],
                **{k: r["primary"][k] for k in ("n_valid", "mae", "rmse", "wape_pct", "wape_denominator")},
                by_client_geo={k: v["wape_pct"] for k, v in r["primary"]["by_client_geo"].items()})


def main():
    ap = argparse.ArgumentParser(prog="python -m aiori2.baseline")
    ap.add_argument("--history", choices=["A", "B"], default="A")
    ap.add_argument("--max-windows", default="1000000")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = evaluate(a.history, None if a.max_windows == "all" else int(a.max_windows))
    print(json.dumps(res, indent=1))
    if a.out:
        core.save_json(a.out, res)


if __name__ == "__main__":
    main()
