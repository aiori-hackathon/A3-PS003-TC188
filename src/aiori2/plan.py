"""Three-stage experiment plan (validation-driven; test data is never used for selection).

Stage 1  model screening: 5 models x {centralised, local, IID FedAvg, Non-IID FedAvg} x {A, B}        = 40 runs
Stage 2  FL comparison: top-2 Stage-1 models x 8 algorithms x {IID, Non-IID} x {A, B}                 = 64 configs,
         of which the 8 FedAvg configs are identical to Stage-1 runs (same run_id) and are not re-run  = 56 new runs
Stage 3  finalists x extra seeds (finalists chosen later from recorded validation results).

    python -m aiori2.plan list|status|estimate|write|run --stage 1 [--config base.json] [--set k=v] [--top mlp,gru]
                     [--finalists RUN_ID,...] [--seeds 1,2] [--only arm=local,model=gru] [--bench FILE] [--dir DIR] [--max-runs N]
"""

import argparse
import copy
import json
import math
from pathlib import Path

from aiori2 import core, track
from aiori2.fl import ALGORITHMS
from aiori2.fl.engine import DEFAULTS as FL_DEFAULTS
from aiori2.models import MODELS
from aiori2.run import FL_ARMS, FL_KEYS, Runner, _parse_set, load_config, run_id

STAGE1_ARMS = [("centralised", None), ("local", None), ("fedavg_iid", "fedavg"), ("fedavg_geo", "fedavg")]


def _cfg(base, **kw):
    c = copy.deepcopy(base)
    c.update(kw)
    return load_config(None, c)


def stage1(base):
    return [_cfg(base, arm=a, algorithm=alg, model=m, history=h) for h in ("A", "B") for m in MODELS for a, alg in STAGE1_ARMS]


def top_models(base, k=2, root=None):
    """Rank models by mean best validation WAPE over their completed Stage-1 runs (all 8 must be finished)."""
    done = {r["run_id"]: r for r in track.collect(root) if r["_status"] in ("completed", "early_stopped")}
    scores = {}
    for m in MODELS:
        runs = [c for c in stage1(base) if c["model"] == m]
        vals = [(done[run_id(c)]["best"].get("val_full") or done[run_id(c)]["best"]["val"])["wape_pct"] for c in runs if run_id(c) in done]
        if len(vals) == len(runs):
            scores[m] = sum(vals) / len(vals)
    if len(scores) < len(MODELS):
        raise RuntimeError(f"Stage 1 incomplete: models with all runs finished: {sorted(scores)}")
    return sorted(scores, key=scores.get)[:k], scores


def alg_fl(base, alg):
    """The run's fl block plus the algorithm's own defaults (configs/fl/defaults.json: e.g. FedProx mu, FedAdam server_lr).
    Without this the run defaults' generic fl values mask them and FedProx/FedAvgM silently train as FedAvg.
    FedAvg has no overrides, so its configs (and the reused Stage-1 run ids) are unchanged."""
    return dict(base["fl"], **{k: v for k, v in FL_DEFAULTS["algorithms"].get(alg, {}).items() if k in FL_KEYS})


def stage2(base, models):
    return [_cfg(base, arm=a, algorithm=alg, model=m, history=h, fl=alg_fl(base, alg))
            for h in ("A", "B") for m in models for a in FL_ARMS for alg in ALGORITHMS]


def stage3(base, finalists, seeds, root=None):
    """Each finalist's recorded config (algorithm settings included) with a new seed, retrained from scratch.
    Budget, output root and device come from `base`, so extended runs get their own run ids and never land in the
    screening output root; `root` is where the finalists' screening records live."""
    by_id = {r["run_id"]: r for r in track.collect(root)}
    out = []
    for fid in finalists:
        if fid not in by_id:
            raise KeyError(f"finalist {fid} has no recorded run")
        for s in seeds:
            out.append(_cfg(by_id[fid]["config"], seed=int(s), budget=copy.deepcopy(base["budget"]),
                            out_root=base["out_root"], device=base["device"]))
    return out


def planned(stage, base, top=None, finalists=None, seeds=None, root=None):
    if stage == 1:
        cfgs = stage1(base)
    elif stage == 2:
        cfgs = stage2(base, top or top_models(base, root=root)[0])
    else:
        cfgs = stage3(base, finalists or [], seeds or [1, 2], root)
    seen, out = set(), []
    for c in cfgs:                                                     # deduplicate by run identity
        rid = run_id(c)
        if rid not in seen:
            seen.add(rid); out.append(c)
    return out


def status_table(cfgs, root=None):
    recorded = {r["run_id"]: r["_status"] for r in track.collect(root)}
    s1 = None
    rows = []
    for c in cfgs:
        rid = run_id(c)
        rows.append(dict(run_id=rid, arm=c["arm"], history=c["history"], model=c["model"], algorithm=c["algorithm"] or "-",
                         status=recorded.get(rid, "missing")))
    return rows


def estimate(cfgs, bench):
    """Hours per config from measured throughput (windows/s); clearly an estimate, not a measurement."""
    b = core.load_json(bench)
    total, rows = 0.0, []
    for c in cfgs:
        kind = "fl" if c["arm"] in FL_ARMS else "centralised"
        thr = b["train_windows_per_s"][c["model"]][kind]
        val = b["val_windows_per_s"][c["model"]]
        e, W = c["budget"]["epochs"], c["budget"]["windows_per_epoch"]
        vw = c["validation"]["max_windows"] or b["val_windows_total"]
        sec = e * W / thr + math.ceil(e / c["validation"]["every_epochs"]) * vw / val
        if c["validation"]["final_full"] and c["validation"]["max_windows"]:
            sec += b["val_windows_total"] / val
        rows.append(dict(run_id=run_id(c), hours_max=round(sec / 3600, 3)))
        total += sec
    return dict(device=b.get("device"), configs=len(cfgs), hours_upper_bound=round(total / 3600, 2), per_config=rows,
                note="upper bound: assumes every run uses all epochs (early stopping shortens runs); measured on " + str(b.get("device")))


def _only(cfgs, spec):
    if not spec:
        return cfgs
    cond = dict(x.split("=", 1) for x in spec.split(","))
    return [c for c in cfgs if all(str(c.get(k) or "none") == v for k, v in cond.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2.plan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "status", "estimate", "write", "run"])
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--config"); ap.add_argument("--set", action="append")
    ap.add_argument("--top"); ap.add_argument("--finalists"); ap.add_argument("--seeds")
    ap.add_argument("--only"); ap.add_argument("--bench"); ap.add_argument("--dir"); ap.add_argument("--max-runs", type=int)
    ap.add_argument("--out-root")
    ap.add_argument("--source-root", help="stage 3: output root holding the finalists' screening runs (default: --out-root)")
    a = ap.parse_args(argv)
    base = load_config(a.config, _parse_set(a.set))
    if a.out_root:
        base["out_root"] = a.out_root
    if a.stage == 3 and a.source_root and Path(a.source_root).resolve() == Path(base["out_root"]).resolve():
        raise SystemExit("stage 3 needs an output root separate from the screening runs")
    cfgs = _only(planned(a.stage, base, a.top.split(",") if a.top else None, a.finalists.split(",") if a.finalists else None,
                         a.seeds.split(",") if a.seeds else None, a.source_root if a.stage == 3 and a.source_root else base["out_root"]),
                 a.only)
    if a.command in ("list", "status"):
        rows = status_table(cfgs, base["out_root"])
        for r in rows:
            print(f"{r['status']:13s} {r['run_id']}" if a.command == "status" else r["run_id"])
        counts = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"stage {a.stage}: {len(rows)} configs {counts}")
    elif a.command == "estimate":
        print(json.dumps({k: v for k, v in estimate(cfgs, a.bench).items() if k != "per_config"}, indent=1))
    elif a.command == "write":
        d = Path(a.dir)
        d.mkdir(parents=True, exist_ok=True)
        for c in cfgs:
            p = d / f"{run_id(c)}.json"
            if not p.exists():
                p.write_text(json.dumps(c, indent=1), encoding="utf-8")
        print(f"{len(cfgs)} configs in {d}")
    else:
        recorded = {r["run_id"]: r["_status"] for r in track.collect(base["out_root"])}
        n = 0
        for c in cfgs:
            st = recorded.get(run_id(c))
            if st in ("completed", "early_stopped"):
                continue
            if a.max_runs is not None and n >= a.max_runs:
                break
            Runner(c).run(resume=st in ("interrupted", "running", "failed"))
            n += 1
        print(f"ran {n} configs")


if __name__ == "__main__":
    main()
