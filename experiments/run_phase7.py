"""Phase 7 federated runner (CPU): FedAvg (geographic or IID-like) and local-only B1 GRU.

Run from the project root:
    .venv\\Scripts\\python.exe experiments/run_phase7.py --config experiments/configs/phase7_fedavg_geo_b1.json --smoke

Validation only: there is no test option, and test windows are not loaded. --smoke runs
2 rounds (or epochs) x 5 batches per client: a pipeline check, not a result.
Outputs go to results/phase7/<run>/<arm>_seed<k>[_smoke]/ (git-ignored).
"""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aiori import dataset, evaluation, federated, training  # noqa: E402
from aiori.model import count_parameters, ForecastGRU  # noqa: E402

log = logging.getLogger("run_phase7")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", choices=sorted(dataset.RUNS), default="primary")
    p.add_argument("--seed", type=int, default=None, help="default: the config's first seed")
    p.add_argument("--smoke", action="store_true", help="2 rounds/epochs x 5 batches per client")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: results/<config results_subdir, else phase7>")
    return p.parse_args(argv)


def make_partition(run, cfg):
    if cfg["partition"] == "geographic":
        return federated.geographic_partition(run)
    if cfg["partition"] == "iid_sensor_balanced":
        return federated.iid_partition(run, cfg["partition_seed"])
    raise ValueError(f"Unknown partition {cfg['partition']!r}")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return o


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = args.config.read_bytes()
    config_sha256 = hashlib.sha256(raw).hexdigest()
    cfg = json.loads(raw)
    if cfg.get("client_fraction", 1.0) != 1.0:
        raise SystemExit("Only full participation (client_fraction 1.0) is implemented")
    out_root = args.out_dir or ROOT / "results" / cfg.get("results_subdir", "phase7")
    seed = cfg["seeds"][0] if args.seed is None else args.seed
    limit = 2 if args.smoke else None
    max_batches = 5 if args.smoke else None
    tag = f"{cfg['arm']}_seed{seed}" + ("_smoke" if args.smoke else "")
    out = out_root / args.run / tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing results in {out}")

    training.configure_torch(cfg["threads"])
    run = dataset.load_run(args.run, splits=("train", "val"))
    partition = make_partition(run, cfg)
    partition_json = {k: v.tolist() for k, v in sorted(partition.items())}
    partition_sha256 = hashlib.sha256(json.dumps(partition_json, sort_keys=True).encode()).hexdigest()

    t = time.perf_counter()
    if "rounds_max" in cfg:
        res = federated.run_fedavg(run, partition, cfg, seed, rounds_max=limit, max_batches=max_batches)
        states = {"global": res["state"]}
        summary = {"best_round": res["best_round"], "best_val_mae": res["best_val_mae"],
                   "early_stopped": res["early_stopped"], "rounds_run": len(res["history"]),
                   "best_pooled_from_sums": res["best_pooled_from_sums"], "n_k": res["n_k"]}
        train_bytes = [b["bytes"] for b in res["byte_log"] if b["purpose"] == "train"]
        summary["communication"] = {
            "payload_bytes_per_model": federated.payload_bytes(res["state"]),
            "raw_float32_bytes_per_model": federated.raw_bytes(res["state"]),
            "train_bytes_per_round": [h["train_bytes"] for h in res["history"]],
            "train_bytes_total": int(sum(train_bytes)),
            "train_bytes_to_best_round": int(sum(b["bytes"] for b in res["byte_log"]
                                                 if b["purpose"] == "train" and b["round"] <= res["best_round"])),
            "eval_download_bytes_total": int(sum(b["bytes"] for b in res["byte_log"] if b["purpose"] == "eval")),
        }
        history = res["history"]
    else:
        res = federated.run_local_only(run, partition, cfg, seed, max_epochs=limit, max_batches=max_batches)
        states = res["states"]
        summary = {name: {"best_epoch": f["best_epoch"], "best_val_mae_own_client": f["best_val_mae"],
                          "early_stopped": f["early_stopped"], "epochs_run": len(f["history"])}
                   for name, f in res["clients"].items()}
        summary = {"clients": summary, "communication": {"train_bytes_total": 0}}
        history = {name: f["history"] for name, f in res["clients"].items()}
    fit_seconds = time.perf_counter() - t

    pred = res["predictions"]
    if not np.isfinite(pred["y_hat_z"]).all():
        raise FloatingPointError("Non-finite predictions")
    if not np.array_equal(np.stack([pred["sensor_pos"], pred["t0"]], 1), run.windows["val"]):
        raise AssertionError("Merged predictions do not cover the validation windows exactly")
    result = evaluation.evaluate_predictions(pred, run)      # harness-side analysis (outside the protocol)
    if "best_pooled_from_sums" in summary and summary["best_pooled_from_sums"]:
        s = summary["best_pooled_from_sums"]
        summary["sums_match_records"] = bool(s["n"] == result["pooled"]["n"]
                                             and np.isclose(s["mae"], result["pooled"]["mae"], rtol=1e-9)
                                             and np.isclose(s["rmse"], result["pooled"]["rmse"], rtol=1e-9))
    finite = all(torch.isfinite(v).all() for st in states.values() for v in st.values())

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg, "config_file": args.config.name, "config_sha256": config_sha256, "run": args.run,
        "seed": seed, "smoke": args.smoke, "max_rounds_or_epochs": limit or cfg.get("rounds_max", cfg.get("max_epochs")),
        "max_batches_per_client_epoch": max_batches, "partition": partition_json, "partition_sha256": partition_sha256,
        "windows_loaded": {k: len(v) for k, v in run.windows.items()}, "evaluated_split": "val",
        "n_parameters": count_parameters(ForecastGRU(4, cfg["hidden_size"])), "weights_finite": bool(finite),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": torch.__version__, "platform": platform.platform()},
        "threads": torch.get_num_threads(), "fit_seconds": fit_seconds,
        "created_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
    }
    report = {"meta": meta, "summary": summary, "history": history, "metrics": result}
    (out / "metrics.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (out / args.config.name).write_bytes(raw)
    (out / "partition.json").write_text(json.dumps(partition_json, indent=2), encoding="utf-8")
    for name, st in states.items():
        torch.save(st, out / f"best_model_{name}.pt")
    evaluation.write_prediction_record(out / f"predictions_{cfg['arm']}_{seed}_val.npz", pred, run, _jsonable(meta))
    log.info("%s %s val pooled MAE %.3f RMSE %.3f (n=%d) in %.1fs; wrote %s", cfg["arm"], args.run,
             result["pooled"]["mae"], result["pooled"]["rmse"], result["pooled"]["n"], fit_seconds, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
