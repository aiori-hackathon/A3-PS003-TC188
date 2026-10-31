"""Phase 9 worker: one task per process, so a crash never takes down the controller.

    python worker.py audit | prepare | verify | smoke | benchstep <out.json> | train <arm> <seed> --deadline <epoch_s>

Every task is idempotent: a completed task (marker present and hashes valid) returns immediately.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiori import config, dataset, evaluation, federated, nine_month as nm, resumable, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402

D = C.R / "data"


def _done(marker):
    m = C.read_json(marker)
    return bool(m) and all(Path(p).is_file() and C.sha256_file(p) == h for p, h in m.get("files", {}).items())


def _mark(marker, files, **extra):
    C.write_json(marker, dict(completed_utc=C.now(), files={str(p): C.sha256_file(p) for p in files}, **extra))


def task_audit(_):
    marker = D / "AUDIT_DONE.json"
    if _done(marker):
        return 0
    paths = nm.month_paths(C.YEAR)
    sp = nm.split_bounds()
    t = time.time()
    df, extra = nm.audit(paths)
    jan = pd.read_csv(config.PROCESSED_DIR / "flow_sensor_audit.csv", usecols=["array_index", "proposed_client", "District"])
    df = nm.eligibility(df).merge(jan, on="array_index", validate="one_to_one")
    D.mkdir(parents=True, exist_ok=True)
    df.to_csv(D / "audit_train_period.csv", index=False)                 # local only (sensor level)
    order = nm.ordering_checks(extra)
    try:
        dst = nm.dst_check(extra["dst_profile"])
    except Exception as e:  # noqa: BLE001 - descriptive check only
        dst = {"error": str(e)}
    summary = dict(months=[p.name for p in paths], split_steps=sp, split_dates=nm.EXPECTED_DATES, sensors=len(df),
                   eligible=int(df["eligible"].sum()), exclusion_counts=df["exclusion_reason"].replace("", "eligible").value_counts().to_dict(),
                   exact_copy_sensors=int(df["duplicate_series"].sum()), ordering=order, dst=dst, seconds=round(time.time() - t, 1),
                   october_december_opened=False)
    C.write_json(D / "audit_summary.json", summary)
    if not order["passed"]:
        C.log(f"AUDIT: sensor-order consistency check FAILED {order}")
        return 3
    _mark(marker, [D / "audit_train_period.csv", D / "audit_summary.json"])
    C.log(f"AUDIT done: {summary['eligible']} eligible of {summary['sensors']}; ordering {order['month_mean_rank_corr']}; dst {dst}")
    return 0


def task_prepare(_):
    marker = D / "PREPARE_DONE.json"
    if _done(marker):
        return 0
    if not _done(D / "AUDIT_DONE.json"):
        raise RuntimeError("audit not complete")
    if C.DATA.exists():                                                     # incomplete earlier attempt: keep it aside
        aside = C.DATA.with_name(C.DATA.name + f"_incomplete_{int(time.time())}")
        C.DATA.rename(aside)
        C.log(f"PREPARE: moved incomplete checkpoint aside to {aside.name}")
    elig = pd.read_csv(D / "audit_train_period.csv", keep_default_na=False, na_values=[""])
    elig["exclusion_reason"] = elig["exclusion_reason"].astype(str).replace("nan", "")
    meta = pd.read_csv(config.PROCESSED_DIR / "sensor_metadata_aligned.csv", usecols=["Type", "District"])
    types = meta["Type"]
    jan = pd.read_csv(config.PROCESSED_DIR / "geo200" / "sensor_selection.csv")
    pilot = pd.concat([pd.read_csv(config.PROCESSED_DIR / f, usecols=["array_index"]) for f in
                       ("sensor_selection.csv", "missing_validation_selection.csv")])["array_index"].tolist()
    sel, repl = nm.select(elig, types, jan, pilot)
    node_order = np.load(config.NODE_ORDER_PATH)
    counts = nm.prepare(nm.month_paths(C.YEAR), sel, node_order, meta["District"].to_numpy(), C.DATA)
    summary = dict(retained_january=int((sel["selection_basis"] == "retained_january").sum()),
                   replacements=pd.Series([r["basis"] for r in repl]).value_counts().to_dict(),
                   replacement_reasons=pd.Series([r["dropped_reason"] for r in repl]).value_counts().to_dict(),
                   per_client=sel["proposed_client"].value_counts().to_dict(), type_groups=sel["type_group"].value_counts().to_dict(),
                   windows=counts, salt=nm.SALT)
    C.write_json(D / "selection_summary.json", summary)
    files = sorted(p for p in C.DATA.iterdir() if p.is_file())
    _mark(marker, files + [D / "selection_summary.json"])
    C.log(f"PREPARE done: {summary}")
    return 0


def task_verify(_):
    marker = D / "VERIFY_DONE.json"
    if _done(marker):
        return 0
    res = nm.verify(C.DATA, nm.month_paths(C.YEAR))
    rows = [dict(check=c, ok=bool(o), detail=str(d)) for c, o, d in res]
    run = dataset.load_run("nine_month", run_dir=C.DATA, selection=C.DATA / "sensor_selection.csv", splits=("train", "val"))
    for split in ("train", "val"):
        ds = dataset.WindowDataset(run, split, calendar=True)
        b = ds.get_batch(np.arange(0, len(ds), max(1, len(ds) // 3000)))
        sp, t0 = b["sensor_pos"].numpy(), b["t0"].numpy()
        nan = np.isnan(run.targets_scaled[sp[:, None], t0[:, None] + np.arange(12)])
        rows.append(dict(check=f"loader {split}: finite x,y and mask == ~NaN", ok=bool(np.isfinite(b["x"].numpy()).all()
                         and np.isfinite(b["y"].numpy()).all() and np.array_equal(b["mask"].numpy(), ~nan)), detail=len(ds)))
    rows.append(dict(check="test windows not loaded by the training loader", ok="test" not in run.windows, detail=""))
    ok = all(r["ok"] for r in rows)
    C.write_json(D / "verification.json", dict(passed=ok, checks=rows, utc=C.now()))
    C.log(f"VERIFY {'PASSED' if ok else 'FAILED'}: {sum(r['ok'] for r in rows)}/{len(rows)}")
    if not ok:
        return 3
    _mark(marker, [D / "verification.json"])
    return 0


def _run():
    return dataset.load_run("nine_month", run_dir=C.DATA, selection=C.DATA / "sensor_selection.csv", splits=("train", "val"))


def _cfg(arm):
    return json.loads((C.ROOT / "experiments" / "configs" / f"phase9_{arm}_b1.json").read_text(encoding="utf-8"))


def _partition(run, arm, cfg):
    return federated.iid_partition(run, cfg["partition_seed"]) if arm == "fedavg_iid" else federated.geographic_partition(run)


def task_smoke(_):
    marker = D / "SMOKE_DONE.json"
    if _done(marker):
        return 0
    run, out = _run(), {}
    sm = C.R / "smoke"
    if sm.exists():
        shutil.rmtree(sm)
    for arm in C.ARMS:
        cfg = dict(_cfg(arm))
        training.configure_torch(cfg["threads"])
        t = time.time()
        if arm == "centralised":
            training.seed_everything(0)
            tr, va = dataset.WindowDataset(run, "train", calendar=True), dataset.WindowDataset(run, "val", calendar=True)
            m = ForecastGRU(tr.n_features, cfg["hidden_size"])
            r = resumable.fit(m, tr, va, run, seed=0, ckpt=sm / arm / "c.pt", max_epochs=2, max_batches=5)
            hist, pred = r["history"], training.predict(m, va)
        elif arm == "local":
            r = resumable.local_only(run, _partition(run, arm, cfg), cfg, 0, ckpt_dir=sm / arm, max_epochs=2, max_batches=5)
            hist, pred = [h for c in r["clients"].values() for h in c["history"]], r["predictions"]
        else:
            r = resumable.fedavg(run, _partition(run, arm, cfg), cfg, 0, ckpt=sm / arm / "f.pt", rounds_max=2, max_batches=5)
            hist, pred = r["history"], r["predictions"]
        met = evaluation.evaluate_predictions(pred, run)["pooled"]
        ok = all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_mae"]) for h in hist) and np.isfinite(pred["y_hat_z"]).all()
        out[arm] = dict(ok=bool(ok), val_mae=met["mae"], seconds=round(time.time() - t, 1))
    C.write_json(D / "smoke.json", out)
    C.log(f"SMOKE {out}")
    if not all(v["ok"] for v in out.values()):
        return 3
    _mark(marker, [D / "smoke.json"])
    return 0


def task_benchstep(args):
    """Time 60 optimiser steps (after 10 warm-up) of the centralised B1 model on real data."""
    cfg = _cfg("centralised")
    training.configure_torch(cfg["threads"])
    run = _run()
    tr = dataset.WindowDataset(run, "train", calendar=True)
    training.seed_everything(0)
    m = ForecastGRU(tr.n_features, cfg["hidden_size"])
    opt = __import__("torch").optim.Adam(m.parameters(), lr=1e-3)
    loader = dataset.make_loader(tr, 512, shuffle=True, seed=0)
    it = iter(loader)
    training.train_one_epoch(m, [next(it) for _ in range(10)], opt, 1.0)
    batches = [next(it) for _ in range(60)]
    t = time.perf_counter()
    training.train_one_epoch(m, batches, opt, 1.0)
    C.write_json(args.out, dict(ms_per_step=1000 * (time.perf_counter() - t) / 60, train_windows=len(tr), peak_mb=C.peak_mb()))
    return 0


def task_train(args):
    arm, seed = args.arm, args.seed
    d = C.RUNS / C.tag(arm, seed)
    if _done(d / "COMPLETE.json"):
        return 0
    cfg = _cfg(arm)
    training.configure_torch(cfg["threads"])
    run = _run()
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    deadline = args.deadline

    def progress(h):
        C.write_json(d / "progress.json", dict(utc=C.now(), unit=h.get("epoch", h.get("round")), val_mae=h["val_mae"]))

    t0 = time.time()
    part = None
    if arm == "centralised":
        training.seed_everything(seed)
        tr, va = dataset.WindowDataset(run, "train", calendar=True), dataset.WindowDataset(run, "val", calendar=True)
        model = ForecastGRU(tr.n_features, cfg["hidden_size"])
        res = resumable.fit(model, tr, va, run, seed=seed, ckpt=d / "ckpt" / "central.pt", lr=cfg["lr"], batch_size=cfg["batch_size"],
                            max_epochs=cfg["max_epochs"], patience=cfg["patience"], grad_clip=cfg["grad_clip"], deadline=deadline,
                            on_epoch=progress)
        pred, states, history = training.predict(model, va), {"all": model.state_dict()}, res["history"]
        summary = dict(best_unit=res["best_epoch"], units_run=len(history), early_stopped=res["early_stopped"], best_val_mae=res["best_val_mae"])
        seconds = sum(h["epoch_seconds"] for h in history)
        comm = dict(train_bytes_total=0)
    elif arm == "local":
        part = _partition(run, arm, cfg)
        res = resumable.local_only(run, part, cfg, seed, ckpt_dir=d / "ckpt", deadline=deadline, on_epoch=progress)
        pred, states = res["predictions"], res["states"]
        history = {c: v["history"] for c, v in res["clients"].items()}
        summary = dict(clients={c: dict(best_epoch=v["best_epoch"], epochs_run=len(v["history"]), early_stopped=v["early_stopped"],
                                        best_val_mae_own=v["best_val_mae"]) for c, v in res["clients"].items()})
        seconds = sum(h["epoch_seconds"] for v in history.values() for h in v)
        comm = dict(train_bytes_total=0)
    else:
        part = _partition(run, arm, cfg)
        res = resumable.fedavg(run, part, cfg, seed, ckpt=d / "ckpt" / "fedavg.pt", deadline=deadline, on_round=progress)
        pred, states, history = res["predictions"], {"global": res["state"]}, res["history"]
        summary = dict(best_unit=res["best_round"], units_run=len(history), early_stopped=res["early_stopped"], best_val_mae=res["best_val_mae"],
                       n_k=res["n_k"])
        seconds = sum(h["round_seconds"] for h in history)
        tb = [b for b in res["byte_log"] if b["purpose"] == "train"]
        comm = dict(payload_bytes_per_model=federated.payload_bytes(res["state"]), train_bytes_total=int(sum(b["bytes"] for b in tb)),
                    train_bytes_to_best=int(sum(b["bytes"] for b in tb if b["round"] <= res["best_round"])),
                    eval_bytes_total=int(sum(b["bytes"] for b in res["byte_log"] if b["purpose"] == "eval")))
    if not np.isfinite(pred["y_hat_z"]).all():
        raise FloatingPointError("non-finite validation predictions")
    if not np.array_equal(np.stack([pred["sensor_pos"], pred["t0"]], 1), run.windows["val"]):
        raise AssertionError("predictions do not cover the validation windows")
    met = evaluation.evaluate_predictions(pred, run)
    import torch
    files = []
    for name, st in states.items():
        p = d / ("best_model.pt" if name == "all" else f"best_model_{name}.pt")
        resumable.atomic_save(st, p)
        files.append(p)
    evaluation.write_prediction_record(d / "predictions_val.npz", pred, run, dict(arm=arm, seed=seed, split="val"))
    if part is not None:
        C.write_json(d / "partition.json", {k: v.tolist() for k, v in sorted(part.items())})
        files.append(d / "partition.json")
    C.write_json(d / "history.json", history)
    C.write_json(d / "metrics_val.json", met)
    C.write_json(d / "run_meta.json", dict(arm=arm, seed=seed, summary=summary, train_seconds_total=seconds,
                                          this_process_seconds=time.time() - t0, peak_mb_this_process=C.peak_mb(),
                                          communication=comm, torch=torch.__version__, threads=cfg["threads"]))
    files += [d / f for f in ("predictions_val.npz", "history.json", "metrics_val.json", "run_meta.json", "config.json")]
    _mark(d / "COMPLETE.json", files, val_mae=met["pooled"]["mae"], val_rmse=met["pooled"]["rmse"])
    C.log(f"TRAIN {C.tag(arm, seed)} complete: val MAE {met['pooled']['mae']:.3f}, {seconds / 60:.1f} min of training")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("arm", nargs="?")
    p.add_argument("seed", nargs="?", type=int)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    fn = {"audit": task_audit, "prepare": task_prepare, "verify": task_verify, "smoke": task_smoke,
          "benchstep": task_benchstep, "train": task_train}[a.task]
    try:
        return fn(a)
    except resumable.Interrupted as e:
        C.log(f"{a.task} {a.arm or ''} {a.seed if a.seed is not None else ''}: interrupted ({e}); checkpoint kept")
        return 4


if __name__ == "__main__":
    sys.exit(main())
