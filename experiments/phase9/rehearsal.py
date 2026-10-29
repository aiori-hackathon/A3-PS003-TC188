"""End-to-end rehearsal of the Phase 9 runner on a synthetic 273-day, 8-sensor dataset (no real data touched).

    python rehearsal.py <workdir> [--deadline-minutes 40] [--kill-controller-after S]

Builds a synthetic checkpoint with the real Phase 9 preprocessing, marks the real-data stages (audit, prepare,
verify) as done, then runs the real controller (smoke, benchmark, 12 training runs, freeze, validation pre-check,
one-time test, analysis, report) with every output redirected to <workdir> via environment variables.
With --kill-controller-after the controller is killed once and restarted, to rehearse crash recovery.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiori import nine_month as nm  # noqa: E402

CLIENTS = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"]


def build(work):
    data = work / "data"
    rng = np.random.default_rng(0)
    n, t = 8, 273 * 288
    steps = np.arange(t)
    daily = 1 + 0.8 * np.sin(2 * np.pi * (steps % 288) / 288 - 1.2)
    flow = 60 * daily[None] * rng.uniform(0.5, 2, (n, 1)) + rng.normal(0, 4, (n, t))
    flow = np.maximum(flow, 0)
    flow[1, 1000:1010] = np.nan
    flow[3, 60000:60004] = np.nan                     # missing validation targets
    flow[5, 70000:70003] = np.nan                     # missing test targets
    sel = pd.DataFrame(dict(array_index=np.arange(n), proposed_client=np.repeat(CLIENTS, 2), type_group="mainline",
                            flow_tertile="mid", selection_basis="synthetic"))
    nm.build_checkpoint(flow, sel, np.arange(900000, 900000 + n), np.repeat([7, 4, 12, 8], 2), data)
    return data


def env(work, data):
    e = dict(os.environ, PHASE9_RESULTS=str(work / "results"), PHASE9_DATA=str(data), PHASE9_DOCS=str(work / "docs"),
             PHASE9_RESERVE_S="240")
    return e


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("--deadline-minutes", type=float, default=40)
    p.add_argument("--kill-controller-after", type=float, default=None)
    a = p.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=False)
    data = build(a.work)
    st = a.work / "results" / "state"
    st.mkdir(parents=True)
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=a.deadline_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stages = {s: {"status": "pending", "attempts": 0} for s in ("audit", "prepare", "verify", "smoke", "benchmark", "train",
                                                                "freeze", "precheck", "test", "analysis", "report")}
    for s in ("audit", "prepare", "verify"):
        stages[s]["status"] = "done"
    (st / "experiment_status.json").write_text(json.dumps(dict(created_utc="rehearsal", deadline_utc=deadline, stages=stages, runs={},
                                                               decisions=["rehearsal: synthetic data; real-data stages marked done"], tasks=[])))
    cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(HERE / "run_nine_month_experiment.py")]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env(a.work, data))
    if a.kill_controller_after:
        time.sleep(a.kill_controller_after)
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        print(f"killed controller tree at {time.time() - t0:.0f}s; restarting")
        time.sleep(3)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env(a.work, data))
    proc.wait()
    s = json.loads((st / "experiment_status.json").read_text())
    print(json.dumps(dict(status=s.get("status"), stages={k: v["status"] for k, v in s["stages"].items()},
                          runs={k: v["status"] for k, v in s["runs"].items()}, minutes=round((time.time() - t0) / 60, 1)), indent=1))
    return 0 if s.get("status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
