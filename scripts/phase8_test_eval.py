"""Phase 8 inference-only evaluation of the frozen S4 checkpoints (docs/phase8_test_protocol.md, E2/E3).

Run from the project root, only after the corresponding gate:
    E2 (validation pre-check, never loads test windows):
        .venv\\Scripts\\python.exe scripts/phase8_test_eval.py --mode precheck --protocol-sha256 <sha> --freeze-sha256 <sha>
    E3 (the single test pass):
        .venv\\Scripts\\python.exe scripts/phase8_test_eval.py --mode test --protocol-sha256 <sha> --freeze-sha256 <sha>

Both modes refuse to run unless the protocol, the freeze manifest, every frozen file and this script match their
recorded SHA-256 values. No optimiser is created and no gradient is computed; model parameters are hashed before
and after prediction. Prediction routing reproduces S4 exactly: centralised = one model on all windows; local-only =
each client's own model on its own sensors; FedAvg = the global model, applied per training-partition client and
merged (the same model on every window). E3 additionally needs a passed E2 log and runs once: it writes
RUN_STARTED before any test window is loaded and refuses if the output folder is not empty.
Outputs stay local and git-ignored (results/phase8/...).
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aiori import config, dataset, evaluation, federated, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402

ARMS = ("centralised", "local", "fedavg_iid", "fedavg_geo")
SEEDS = (0, 1, 2)
EXPECTED_TEST_WINDOWS = 343246


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("precheck", "test"), required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--root", type=Path, default=ROOT, help="project root holding results/, docs/ and the frozen code")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR / "geo200")
    p.add_argument("--protocol", type=Path, default=None, help="default: <root>/docs/phase8_test_protocol.md")
    p.add_argument("--expected-test-windows", type=int, default=EXPECTED_TEST_WINDOWS)
    return p.parse_args(argv)


def paths(args):
    pr = args.root / "results" / "phase8"
    return dict(runs=pr / "geo200", freeze=pr / "test_protocol" / "frozen_checkpoints.json",
                e1=pr / "test_protocol" / "e1_record.json", precheck=pr / "test_protocol" / "precheck_log.json",
                out=pr / "test", protocol=args.protocol or args.root / "docs" / "phase8_test_protocol.md")


def verify_frozen(args, P):
    """Fail closed unless the protocol, freeze manifest, every frozen file and this script match their hashes."""
    if sha256_file(P["protocol"]) != args.protocol_sha256:
        raise SystemExit("Refusing: protocol document does not match --protocol-sha256")
    if sha256_file(P["freeze"]) != args.freeze_sha256:
        raise SystemExit("Refusing: freeze manifest does not match --freeze-sha256")
    e1 = json.loads(P["e1"].read_text(encoding="utf-8"))
    if e1["script_sha256"] != sha256_file(__file__):
        raise SystemExit("Refusing: this evaluator differs from the hash recorded at E1")
    if (e1["protocol_sha256"], e1["freeze_sha256"]) != (args.protocol_sha256, args.freeze_sha256):
        raise SystemExit("Refusing: E1 record was made for a different protocol or freeze manifest")
    fz = json.loads(P["freeze"].read_text(encoding="utf-8"))
    expected = {P["runs"] / run / name: h for run, files in fz["runs"].items() for name, h in files.items()}
    expected |= {args.data_dir / name: h for name, h in fz["geo200_artifacts"].items()}
    expected |= {args.data_dir / "checkpoint_manifest.json": fz["geo200_checkpoint_manifest_sha256"],
                 args.root / "results" / "phase8" / "analysis" / "sesoi_log.json": fz["sesoi_log_sha256"]}
    expected |= {args.root / name: h for name, h in fz["code_sha256"].items()}
    expected |= {args.root / "experiments" / "configs" / name: h for name, h in fz["configs_sha256"].items()}
    bad = sorted(str(p) for p, h in expected.items() if not p.is_file() or sha256_file(p) != h)
    if bad:
        raise SystemExit(f"Refusing: {len(bad)} frozen file(s) missing or changed, e.g. {bad[:3]}")
    tags = sorted(fz["runs"])
    want = sorted(f"{a}_b1_geo200_seed{s}" for a in ARMS for s in SEEDS)
    if tags != want:
        raise SystemExit(f"Refusing: freeze manifest does not hold exactly the 12 Phase 8 runs: {tags}")
    return fz, tags, len(expected)


def load_run_spec(run_dir):
    """Config, partition (None for centralised) and checkpoint states keyed by client or 'global'/'all'."""
    cfg = json.loads(next(run_dir.glob("phase8_*_b1.json")).read_text(encoding="utf-8"))
    pts = sorted(run_dir.glob("*.pt"))
    if (run_dir / "partition.json").exists():
        part = {k: np.asarray(v, np.int64) for k, v in json.loads((run_dir / "partition.json").read_text()).items()}
        if [p.name for p in pts] == ["best_model_global.pt"]:
            states = {"global": torch.load(pts[0], weights_only=True)}
        elif sorted(p.name for p in pts) == sorted(f"best_model_{c}.pt" for c in part):
            states = {c: torch.load(run_dir / f"best_model_{c}.pt", weights_only=True) for c in part}
        else:
            raise SystemExit(f"Refusing: unexpected checkpoints in {run_dir.name}: {[p.name for p in pts]}")
    else:
        if [p.name for p in pts] != ["best_model.pt"]:
            raise SystemExit(f"Refusing: unexpected checkpoints in {run_dir.name}: {[p.name for p in pts]}")
        part, states = None, {"all": torch.load(pts[0], weights_only=True)}
    return cfg, part, states


def _model(state, n_features, hidden):
    model = ForecastGRU(n_features, hidden)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_split(run, split, cfg, part, states):
    """Predictions for every `split` window, routed exactly as in S4; parameters must be unchanged afterwards."""
    before = {k: state_hash(v) for k, v in states.items()}
    used = []                                            # (state key, model) pairs, checked after prediction
    with torch.no_grad():
        if part is None:
            ds = dataset.WindowDataset(run, split, calendar=cfg["calendar"])
            model = _model(states["all"], ds.n_features, cfg["hidden_size"])
            used.append(("all", model))
            pred = training.predict(model, ds)
        else:
            preds = []
            for name, positions in sorted(part.items()):
                pos = np.sort(positions)
                sub = federated.subset_run(run, pos, name=f"{run.name}:{name}")
                ds = dataset.WindowDataset(sub, split, calendar=cfg["calendar"])
                key = "global" if "global" in states else name
                model = _model(states[key], ds.n_features, cfg["hidden_size"])
                used.append((key, model))
                p = training.predict(model, ds)
                p["sensor_pos"] = pos[p["sensor_pos"]]
                preds.append(p)
            pred = federated.merge_predictions(preds)
    if any(state_hash(m.state_dict()) != before[k] for k, m in used) or \
            {k: state_hash(v) for k, v in states.items()} != before:
        raise AssertionError("Model parameters changed during prediction")
    if not np.isfinite(pred["y_hat_z"]).all():
        raise FloatingPointError("Non-finite predictions")
    if not np.array_equal(np.stack([pred["sensor_pos"], pred["t0"]], 1), run.windows[split]):
        raise AssertionError(f"Predictions do not cover the {split} windows exactly")
    return pred, before


def _load(args, splits):
    return dataset.load_run("geo200", run_dir=args.data_dir, selection=args.data_dir / "sensor_selection.csv", splits=splits)


def precheck(args, P, tags):
    """E2: re-predict validation from the frozen checkpoints; must equal the saved S4 records byte for byte."""
    if P["precheck"].exists():
        raise SystemExit(f"Refusing to overwrite {P['precheck']}")
    run = _load(args, ("val",))
    rows, ok = {}, True
    for tag in tags:
        cfg, part, states = load_run_spec(P["runs"] / tag)
        training.configure_torch(cfg["threads"])
        pred, hashes = predict_split(run, "val", cfg, part, states)
        with np.load(next((P["runs"] / tag).glob("predictions_*_val.npz"))) as z:
            same = {k: bool(np.array_equal(pred[k], z[k])) for k in ("sensor_pos", "t0", "mask")}
            same["y_hat_z_bytes"] = pred["y_hat_z"].astype(np.float32).tobytes() == z["y_hat_z"].tobytes()
            diff = float(np.abs(pred["y_hat_z"].astype(np.float64) - z["y_hat_z"]).max())
        rows[tag] = dict(identical=all(same.values()), checks=same, max_abs_diff_z=diff, n_windows=int(len(pred["t0"])),
                         models=len(states), param_sha256=hashes)
        ok &= rows[tag]["identical"]
    log = dict(stage="E2 validation pre-check", created_utc=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
               passed=bool(ok), protocol_sha256=args.protocol_sha256, freeze_sha256=args.freeze_sha256,
               script_sha256=sha256_file(__file__), windows_loaded=sorted(run.windows), runs=rows)
    P["precheck"].write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"E2 pre-check {'PASSED' if ok else 'FAILED'}: {sum(r['identical'] for r in rows.values())}/{len(rows)} runs identical")
    return 0 if ok else 1


def test_pass(args, P, tags):
    """E3: the single test pass. Writes RUN_STARTED before any test window is loaded."""
    if not P["precheck"].exists():
        raise SystemExit("Refusing: no E2 pre-check log")
    pc = json.loads(P["precheck"].read_text(encoding="utf-8"))
    if not pc["passed"] or (pc["protocol_sha256"], pc["freeze_sha256"], pc["script_sha256"]) != (
            args.protocol_sha256, args.freeze_sha256, sha256_file(__file__)):
        raise SystemExit("Refusing: E2 pre-check did not pass for this protocol, freeze manifest and evaluator")
    out = P["out"]
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing: {out} is not empty (the test pass runs once)")
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    (out / "RUN_STARTED").write_text(json.dumps(dict(utc=f"{started:%Y-%m-%d %H:%M:%S}", protocol_sha256=args.protocol_sha256,
                                                     freeze_sha256=args.freeze_sha256)), encoding="utf-8")
    run = _load(args, ("test",))
    if len(run.windows["test"]) != args.expected_test_windows:
        raise AssertionError(f"{len(run.windows['test'])} test windows, expected {args.expected_test_windows}")
    key, summary = None, {}
    for tag in tags:
        cfg, part, states = load_run_spec(P["runs"] / tag)
        training.configure_torch(cfg["threads"])
        pred, hashes = predict_split(run, "test", cfg, part, states)
        k = hashlib.sha256(pred["sensor_pos"].tobytes() + pred["t0"].tobytes() + pred["mask"].tobytes()).hexdigest()
        key = key or k
        if k != key:
            raise AssertionError(f"{tag}: test windows or masks differ from the other runs")
        result = evaluation.evaluate_predictions(pred, run)
        meta = dict(run=tag, arm=cfg["arm"], seed=int(tag[-1]), split="test", n_models=len(states), param_sha256=hashes,
                    checkpoint_sha256={n: h for n, h in json.loads(P["freeze"].read_text())["runs"][tag].items() if n.endswith(".pt")},
                    protocol_sha256=args.protocol_sha256, freeze_sha256=args.freeze_sha256)
        d = out / tag
        d.mkdir()
        evaluation.write_prediction_record(d / f"predictions_{cfg['arm']}_{tag[-1]}_test.npz", pred, run, meta)
        (d / "metrics_test.json").write_text(json.dumps(dict(meta=meta, metrics=result), indent=2, default=float), encoding="utf-8")
        summary[tag] = dict(mae=result["pooled"]["mae"], rmse=result["pooled"]["rmse"], n=result["pooled"]["n"])
    (out / "reference_forecasts_test.json").write_text(
        json.dumps(evaluation.evaluate_baselines(run, "test"), indent=2, default=float), encoding="utf-8")
    log = dict(stage="E3 test pass", started_utc=f"{started:%Y-%m-%d %H:%M:%S}",
               finished_utc=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}", windows=int(len(run.windows["test"])),
               windows_loaded=sorted(run.windows), runs=summary, script_sha256=sha256_file(__file__),
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=float), encoding="utf-8")
    (out / "RUN_COMPLETED").write_text(log["finished_utc"], encoding="utf-8")
    print(f"E3 test pass complete: {len(summary)} runs, {log['windows']} windows; wrote {out}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    P = paths(args)
    _, tags, n_files = verify_frozen(args, P)
    print(f"verified {n_files} frozen files and the evaluator hash")
    return precheck(args, P, tags) if args.mode == "precheck" else test_pass(args, P, tags)


if __name__ == "__main__":
    sys.exit(main())
