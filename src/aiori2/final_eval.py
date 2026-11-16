"""Frozen one-time final evaluation (Phase 8C). Training and model selection never import this module.

    python -m aiori2.final_eval freeze   --selection SEL.json --manifest MANIFEST.json
    python -m aiori2.final_eval evaluate --manifest MANIFEST.json --sha256 HASH --out RESULT.json [--split val|test]

freeze: SEL.json lists {"models": [{"label", "run_dir"}...], "persistence": ["A", ...], "policy": {...}}. Every
checkpoint, config and scaler file is hashed into MANIFEST.json (never overwritten). evaluate: refuses unless the
manifest hashes to HASH and every file still matches it, refuses to overwrite RESULT.json (one evaluation per
manifest), then scores each frozen best.pt and the persistence baseline on the full split with the validation
metric code (same masks, raw-flow WAPE/MAE/RMSE, per client and per horizon). --split val reproduces the recorded
full-validation numbers and is how this path is checked without touching test data.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from aiori2 import core
from aiori2.baseline import persistence
from aiori2.run import Evaluator, Runner, arms, code_hash, load_config

KEYS = ("n_valid", "mae", "rmse", "wape_pct", "wape_denominator")


def _metrics(r):
    p = r["primary"]
    return dict(**{k: p[k] for k in KEYS}, windows=r["windows"],
                by_client_geo={c: {k: v[k] for k in KEYS} for c, v in p["by_client_geo"].items()},
                by_client_iid={c: {k: v[k] for k in KEYS} for c, v in p["by_client_iid"].items()},
                by_horizon=p["by_horizon"], per_sensor_wape_pct=p["per_sensor_wape_pct"])


def freeze(selection, manifest, root=None):
    manifest = Path(manifest)
    if manifest.exists():
        raise FileExistsError(f"{manifest} is frozen; refusing to overwrite it")
    sel = core.load_json(selection)
    root = Path(root) if root else core.load_config()["_out"]
    models = []
    for m in sel["models"]:
        d = Path(m["run_dir"])
        info = core.load_json(d / "run.json")
        if info["status"] not in ("completed", "early_stopped"):
            raise ValueError(f"{d} is not a finished run")
        c = info["config"]
        exp = root / "experiments" / c["policy"] / c["history"]
        models.append(dict(label=m["label"], run_id=info["run_id"], run_dir=str(d), arm=c["arm"], history=c["history"], model=c["model"],
                           algorithm=c["algorithm"], seed=c["seed"], best_epoch=info["best"]["epoch"],
                           val_full=info["best"].get("val_full"), config_sha256=info["config_sha256"], code_sha256=info["code_sha256"],
                           best_pt_sha256=core.sha256_file(d / "best.pt"), run_json_sha256=core.sha256_file(d / "run.json"),
                           scaler_scope=arms.ARMS[c["arm"]]["scaler"], scalers_sha256=core.sha256_file(exp / "scalers.json"),
                           test_windows_sha256=core.sha256_file(exp / "windows" / "test.npy"),
                           valid_target_mask_sha256=core.sha256_file(root / "masks" / c["policy"] / "valid_target.npy")))
    out = dict(frozen=time.strftime("%Y-%m-%dT%H:%M:%S"), policy=sel.get("policy"), models=models,
               persistence=[dict(history=h) for h in sel.get("persistence", [])],
               data_manifest_sha256=core.sha256_file(root / "manifests" / "dgx_transfer_manifest.json"), eval_code_sha256=code_hash(),
               metric="raw-flow MAE/RMSE/WAPE over valid targets (policy mask), pooled sums, all windows of the split, per geo/IID client and horizon")
    core.save_json(manifest, out)
    return core.sha256_file(manifest)


def evaluate(manifest, expect_sha, out, split="test", device="cuda", root=None):
    manifest, out = Path(manifest), Path(out)
    if core.sha256_file(manifest) != expect_sha:
        raise ValueError("manifest does not match the frozen hash")
    if out.exists():
        raise FileExistsError(f"{out} exists: this manifest has already been evaluated")
    man = core.load_json(manifest)
    root = Path(root) if root else core.load_config()["_out"]
    if core.sha256_file(root / "manifests" / "dgx_transfer_manifest.json") != man["data_manifest_sha256"]:
        raise ValueError("data manifest changed since freezing")
    for m in man["models"]:                                           # every frozen file must be untouched
        d = Path(m["run_dir"])
        if core.sha256_file(d / "best.pt") != m["best_pt_sha256"] or core.sha256_file(d / "run.json") != m["run_json_sha256"]:
            raise ValueError(f"{d} changed since freezing")
    dev = torch.device(device)
    res = dict(manifest=str(manifest), manifest_sha256=expect_sha, split=split, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               eval_code_sha256=code_hash(), models=[], persistence=[])
    for m in man["models"]:
        d = Path(m["run_dir"])
        cfg = load_config(None, dict(core.load_json(d / "config.json"), device=device))
        r = Runner(cfg, root=root, out_root=d.parent)
        if r.id != m["run_id"]:
            raise ValueError(f"{d}: config no longer maps to its run id")
        r._setup()
        meta = r._load_state(d / "best.pt")                            # checks config hash and run id
        ev = Evaluator(root, cfg, None, split).evaluate(lambda X, s: r.predict(X, s), dev)
        res["models"].append(dict(label=m["label"], run_id=m["run_id"], checkpoint_epoch=meta["epoch"], **_metrics(ev)))
        print(f"{m['label']:40s} {split} WAPE {ev['primary']['wape_pct']:.4f}", flush=True)
        del r
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    for p in man["persistence"]:
        cfg = load_config(None, dict(arm="centralised", history=p["history"]))
        ev = Evaluator(root, cfg, None, split).evaluate(persistence, torch.device("cpu"))
        res["persistence"].append(dict(label=f"persistence-{p['history']}", history=p["history"], **_metrics(ev)))
        print(f"persistence-{p['history']:30s} {split} WAPE {ev['primary']['wape_pct']:.4f}", flush=True)
    res["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    res["same_target_set"] = len({x["n_valid"] for x in res["models"] + res["persistence"]}) == 1
    core.save_json(out, res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2.final_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["freeze", "evaluate"])
    ap.add_argument("--selection"); ap.add_argument("--manifest", required=True); ap.add_argument("--sha256")
    ap.add_argument("--out"); ap.add_argument("--split", choices=["val", "test"], default="test"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.command == "freeze":
        print(freeze(a.selection, a.manifest))
    else:
        evaluate(a.manifest, a.sha256, a.out, a.split, a.device)


if __name__ == "__main__":
    main()
