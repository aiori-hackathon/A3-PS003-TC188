"""Phase 6 centralised GRU runner (CPU).

Run from the project root:
    .venv\\Scripts\\python.exe experiments/run_phase6.py --config experiments/configs/phase6_b0.json --smoke

--smoke trains for a few batches only (pipeline check, not a result). Evaluation defaults to
the validation split. The test split is refused unless --final is given together with the
SHA-256 of the exact config file (--approved-config-sha256), i.e. only for an approved,
frozen configuration. Outputs go to results/<config results_subdir, else phase6>/ (git-ignored);
an existing non-empty run folder is never overwritten.
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

from aiori import dataset, evaluation, model as model_lib, training  # noqa: E402

log = logging.getLogger("run_phase6")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", choices=sorted(dataset.RUNS), default="primary")
    p.add_argument("--seed", type=int, default=None, help="default: the config's first seed")
    p.add_argument("--smoke", action="store_true", help="few batches and epochs; pipeline check only")
    p.add_argument("--max-batches", type=int, default=None, help="training batches per epoch (smoke default 20)")
    p.add_argument("--epochs", type=int, default=None, help="override max_epochs (smoke default 2)")
    p.add_argument("--evaluate", choices=("val", "test"), default="val")
    p.add_argument("--final", action="store_true")
    p.add_argument("--approved-config-sha256", default=None)
    p.add_argument("--baselines", action="store_true", help="also score the reference forecasts on the evaluated split")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: results/<config results_subdir, else phase6>")
    return p.parse_args(argv)


def check_test_access(args, config_sha256):
    """Refuse test evaluation unless this is an explicitly approved final run of this exact config."""
    if args.evaluate != "test":
        return
    if args.smoke or not args.final or args.approved_config_sha256 != config_sha256:
        raise SystemExit("Refusing to evaluate the test split: needs --final, no --smoke, and "
                         "--approved-config-sha256 equal to the config file's SHA-256.")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = args.config.read_bytes()
    config_sha256 = hashlib.sha256(raw).hexdigest()
    check_test_access(args, config_sha256)
    cfg = json.loads(raw)
    seed = cfg["seeds"][0] if args.seed is None else args.seed
    epochs = args.epochs or (2 if args.smoke else cfg["max_epochs"])
    max_batches = args.max_batches or (20 if args.smoke else None)
    out_root = args.out_dir or ROOT / "results" / cfg.get("results_subdir", "phase6")
    tag = f"{cfg['arm']}_seed{seed}" + ("_smoke" if args.smoke else "")
    out = out_root / args.run / tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing results in {out}")

    training.configure_torch(cfg["threads"])
    training.seed_everything(seed)
    run = dataset.load_run(args.run, splits=("train", "val") if args.evaluate == "val" else None)
    train_ds = dataset.WindowDataset(run, "train", calendar=cfg["calendar"])
    val_ds = dataset.WindowDataset(run, "val", calendar=cfg["calendar"])
    model = model_lib.ForecastGRU(train_ds.n_features, cfg["hidden_size"])
    log.info("%s on %s: %d parameters, seed %d, %d threads", cfg["arm"], args.run,
             model_lib.count_parameters(model), seed, torch.get_num_threads())

    t = time.perf_counter()
    fit = training.fit(model, train_ds, val_ds, run, seed=seed, lr=cfg["lr"], batch_size=cfg["batch_size"],
                       max_epochs=epochs, patience=cfg["patience"], grad_clip=cfg["grad_clip"],
                       max_batches=max_batches)
    fit_seconds = time.perf_counter() - t

    eval_ds = val_ds if args.evaluate == "val" else dataset.WindowDataset(run, "test", calendar=cfg["calendar"])
    t = time.perf_counter()
    pred = training.predict(model, eval_ds)
    if not np.isfinite(pred["y_hat_z"]).all():
        raise FloatingPointError("Non-finite predictions")
    result = evaluation.evaluate_predictions(pred, run)
    eval_seconds = time.perf_counter() - t

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg, "config_file": args.config.name, "config_sha256": config_sha256, "run": args.run,
        "seed": seed, "smoke": args.smoke, "epochs_allowed": epochs, "max_batches_per_epoch": max_batches,
        "best_epoch": fit["best_epoch"], "best_val_mae": fit["best_val_mae"], "early_stopped": fit["early_stopped"],
        "epochs_run": len(fit["history"]), "evaluated_split": args.evaluate,
        "windows_loaded": {k: len(v) for k, v in run.windows.items()},
        "n_parameters": model_lib.count_parameters(model), "torch_version": torch.__version__,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": torch.__version__, "platform": platform.platform()},
        "threads": torch.get_num_threads(), "fit_seconds": fit_seconds, "eval_seconds": eval_seconds,
        "created_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
    }
    report = {"meta": meta, "history": fit["history"], "metrics": result}
    if args.baselines:
        report["reference_forecasts"] = evaluation.evaluate_baselines(run, args.evaluate)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / args.config.name).write_bytes(raw)                       # exact config used
    torch.save(model.state_dict(), out / "best_model.pt")           # best-epoch weights (restored by fit)
    evaluation.write_prediction_record(out / f"predictions_{cfg['arm']}_{seed}_{args.evaluate}.npz", pred, run, meta)
    log.info("%s %s pooled MAE %.3f RMSE %.3f (n=%d); wrote %s", args.run, args.evaluate,
             result["pooled"]["mae"], result["pooled"]["rmse"], result["pooled"]["n"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
