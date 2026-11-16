"""Experiment tracking: read every <out_root>/<run_id>/run.json and export comparison tables (only trained runs).

    python -m aiori2.track summary [--out-root DIR] [--csv FILE]
    python -m aiori2.track compare-ab [--out-root DIR]          # A vs B best validation WAPE side by side
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from aiori2 import core

STALE_MINUTES = 15


def out_root(path=None):
    p = Path(path or "results/aiori2/runs")
    return p if p.is_absolute() else core.SRC / p


def status(info, now=None):
    """Recorded status, with a 'running' run whose heartbeat is older than STALE_MINUTES reported as interrupted."""
    s = info.get("status")
    if s == "running" and info.get("heartbeat"):
        age = ((now or datetime.now()) - datetime.fromisoformat(info["heartbeat"])).total_seconds() / 60
        if age > STALE_MINUTES:
            return "interrupted"
    return s


def collect(root=None):
    runs = []
    for p in sorted(out_root(root).glob("*/run.json")):
        info = core.load_json(p)
        info["_status"] = status(info)
        runs.append(info)
    return runs


def summary(root=None):
    rows = []
    for r in collect(root):
        c, b, k = r["config"], r.get("best") or {}, r.get("counters", {})
        v = b.get("val_full") or b.get("val") or {}
        rows.append(dict(run_id=r["run_id"], status=r["_status"], arm=c["arm"], history=c["history"], model=c["model"], preset=c["preset"],
                         algorithm=c["algorithm"] or "-", seed=c["seed"], epochs_done=len(r.get("history", [])), best_epoch=b.get("epoch"),
                         val_scope="full" if b.get("val_full") else ("subset" if v else None),
                         val_wape_pct=v.get("wape_pct"), val_mae=v.get("mae"), val_rmse=v.get("rmse"), val_n_valid=v.get("n_valid"),
                         windows=k.get("windows"), steps=k.get("steps"), rounds=k.get("rounds"), train_seconds=round(k.get("train_seconds", 0), 1),
                         fl_mbytes=round((k.get("upload_bytes", 0) + k.get("download_bytes", 0)) / 1e6, 2) or None,
                         peak_gpu_mb=r.get("peak_gpu_mb"), parameters=r.get("parameters"), failure=r.get("failure")))
    return pd.DataFrame(rows)


def compare_ab(root=None):
    df = summary(root)
    if df.empty:
        return df
    df = df[df.status.isin(["completed", "early_stopped"])]
    return df.pivot_table(index=["arm", "model", "algorithm", "seed"], columns="history", values="val_wape_pct", aggfunc="first")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2.track")
    ap.add_argument("command", choices=["summary", "compare-ab"])
    ap.add_argument("--out-root"); ap.add_argument("--csv")
    a = ap.parse_args(argv)
    df = summary(a.out_root) if a.command == "summary" else compare_ab(a.out_root)
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_columns", 30):
        print(df if not df.empty else "no runs recorded")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.csv, index=a.command == "compare-ab")


if __name__ == "__main__":
    main()
