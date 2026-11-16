"""AIORI 2.0 experiment runner: one run = one (arm, history, model, algorithm, seed, settings).

    python -m aiori2.run --config cfg.json [--set key=value ...] [--dry-run] [--resume] [--stop-after-epochs N]

Arms: centralised (one model, pooled scaler), local (four independent geographic models, client scalers),
fedavg_iid / fedavg_geo (Phase 5 FL engine, `algorithm` required). Budget is counted in sampled training windows:
every arm processes `windows_per_epoch` windows per epoch in total (local: split evenly over the four clients;
FL: ceil(W / (clients x local_steps x batch)) rounds per epoch), so A and B and all arms get the same number of
sampled windows and similar optimizer-step counts. Validation (never test) uses the same window population for
every arm and history. Outputs go to <out_root>/<run_id>/ (run.json, config.json, metrics.jsonl, latest.pt, best.pt).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aiori2 import arms, core, pipeline
from aiori2.data import ThreeYearWindows
from aiori2.fl import ALGORITHMS, FL_ARMS, FederatedRun, make_config
from aiori2.losses import batch_tensors, train_step
from aiori2.models import MODELS, build_model, count_parameters, presets

DEFAULTS = core.load_json(Path(__file__).resolve().parent / "configs" / "runs" / "defaults.json")
FL_KEYS = ("local_steps", "client_optimizer", "client_lr", "client_momentum", "server_lr", "mu", "server_momentum", "beta1", "beta2",
           "tau", "participation", "weighting")
STATUS_DONE = ("completed", "early_stopped")


# ---------------------------------------------------------------- configuration

def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy({k: v for k, v in DEFAULTS.items() if k != "note"})
    if path:
        cfg = _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    return validate(_merge(cfg, overrides or {}))


def fl_config(cfg):
    """The Phase 5 engine configuration for an FL run (validated by aiori2.fl)."""
    f = cfg["fl"]
    return make_config(algorithm=cfg["algorithm"], arm=cfg["arm"], history=cfg["history"], policy=cfg["policy"], model=cfg["model"],
                       preset=cfg["preset"], in_channels=cfg["in_channels"], batch_size=cfg["batch_size"], grad_clip=cfg["grad_clip"],
                       loss=cfg["loss"], seed=cfg["seed"], **{k: f[k] for k in FL_KEYS})


def validate(cfg):
    arm = cfg["arm"]
    if arm not in arms.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if cfg["history"] not in arms.HISTORIES:
        raise ValueError("history must be A or B")
    if cfg["model"] not in MODELS or cfg["preset"] not in presets()[cfg["model"]]:
        raise ValueError("unknown model or preset")
    if arm in FL_ARMS:
        if cfg["algorithm"] not in ALGORITHMS:
            raise ValueError(f"arm {arm} needs an FL algorithm from {ALGORITHMS}")
        if cfg["weight_decay"]:
            raise ValueError("weight decay is not supported by the FL engine; set weight_decay=0")
        fl_config(cfg)                                            # raises on unsupported combinations (e.g. SCAFFOLD + Adam)
    elif cfg["algorithm"] is not None:
        raise ValueError(f"arm {arm} is not federated; algorithm must be null")
    if cfg["optimizer"] not in ("adam", "adamw", "sgd"):
        raise ValueError("optimizer must be adam, adamw or sgd")
    b, v, e = cfg["budget"], cfg["validation"], cfg["early_stopping"]
    if not (int(b["epochs"]) >= 1 and int(b["windows_per_epoch"]) >= cfg["batch_size"] >= 1):
        raise ValueError("budget needs epochs >= 1 and windows_per_epoch >= batch_size >= 1")
    if not (v["every_epochs"] >= 1 and v["batch_size"] >= 1 and (v["max_windows"] is None or v["max_windows"] >= 1)):
        raise ValueError("invalid validation settings")
    if e["patience"] < 1 or cfg["lr"] <= 0 or cfg["policy"] not in ("conservative", "outage"):
        raise ValueError("invalid early-stopping, learning-rate or policy setting")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return cfg


def identity(cfg):
    """Fields that define the experiment (device and output location excluded so a run can resume elsewhere)."""
    return {k: v for k, v in cfg.items() if k not in ("device", "out_root")}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(identity(cfg), sort_keys=True).encode()).hexdigest()


def run_id(cfg):
    return f"{cfg['arm']}-{cfg['history']}-{cfg['model']}-{cfg['preset']}-{cfg['algorithm'] or 'none'}-s{cfg['seed']}-{config_hash(cfg)[:10]}"


def code_hash():
    h = hashlib.sha256()
    for p in sorted((core.SRC / "aiori2").rglob("*")):
        if p.suffix in (".py", ".json") and "__pycache__" not in p.parts and "experiments" not in p.parts:
            h.update(p.relative_to(core.SRC).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def _seed(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def environment(device):
    env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_build=torch.version.cuda, platform=platform.platform(),
               cpu=platform.processor(), device=str(device))
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_total_mb=round(p.total_memory / 2**20), capability=list(torch.cuda.get_device_capability(0)))
    return env


# ---------------------------------------------------------------- evaluation (validation only)

def sums_to_metrics(abs_s, sq_s, y_s, n):
    return dict(n_valid=int(n), mae=abs_s / n if n else float("nan"), rmse=math.sqrt(sq_s / n) if n else float("nan"),
                wape_pct=100 * abs_s / y_s if y_s else float("nan"), wape_denominator=float(y_s))


class Evaluator:
    """Streams validation windows in bounded batches and accumulates raw-flow error sums (never per-batch WAPE).
    The window subset is drawn from the shared validation index with a fixed seed, so every arm and both histories
    are scored on the same (sensor, time) population."""

    def __init__(self, root, cfg, max_windows, split="val"):
        """split="test" is only for the frozen one-time final evaluation (aiori2.final_eval); training never passes it."""
        if split not in ("val", "test"):
            raise ValueError("Evaluator scores the val or test split only")
        self.root, self.cfg, self.split = Path(root), cfg, split
        scope = arms.ARMS[cfg["arm"]]["scaler"]
        self.ld = ThreeYearWindows(root, cfg["policy"], cfg["history"], split, scope)
        self.ld2 = (ThreeYearWindows(root, cfg["policy"], cfg["history"], split, scope,
                                     eval_mask=self.root / "phase3" / "sensitivity" / "constant_run_24h" / "keep_mask.npy")
                    if cfg["validation"]["sensitivity_mask"] else None)
        n = len(self.ld)
        self.positions = (np.arange(n) if max_windows is None or max_windows >= n
                          else np.sort(np.random.default_rng(0).choice(n, size=int(max_windows), replace=False)))
        s = pd.read_csv(self.root / "metadata" / "sensors.csv")
        self.client_geo = s["client_geo"].to_numpy()
        self.client_iid = s["client_iid"].to_numpy()

    @torch.no_grad()
    def evaluate(self, predict, device):
        """predict(X tensor, sensor np array) -> scaled (B, 12) tensor. Returns pooled, per-client and per-sensor metrics."""
        acc = {m: [np.zeros(200) for _ in range(4)] for m in (("primary", "sensitivity") if self.ld2 else ("primary",))}
        hz = [np.zeros(12) for _ in range(4)]                                   # primary-mask sums per forecast step
        bs = self.cfg["validation"]["batch_size"]
        t0 = time.perf_counter()
        for j in range(0, len(self.positions), bs):
            idx = self.positions[j:j + bs]
            X, Y, M, s = self.ld.get(idx)
            z = predict(torch.as_tensor(X, device=device), s).float().cpu().numpy()
            pred = self.ld.inverse(z, s)
            masks = {"primary": M}
            if self.ld2:
                masks["sensitivity"] = self.ld2.get(idx)[2]
            for name, m in masks.items():
                e = np.where(m, pred - np.nan_to_num(Y), 0.0)
                y = np.where(m, np.abs(np.nan_to_num(Y)), 0.0)
                a = acc[name]
                a[0] += np.bincount(s, np.abs(e).sum(1), 200); a[1] += np.bincount(s, (e * e).sum(1), 200)
                a[2] += np.bincount(s, y.sum(1), 200); a[3] += np.bincount(s, m.sum(1), 200)
                if name == "primary":
                    hz[0] += np.abs(e).sum(0); hz[1] += (e * e).sum(0); hz[2] += y.sum(0); hz[3] += m.sum(0)
        out = dict(windows=int(len(self.positions)), seconds=round(time.perf_counter() - t0, 2), split=self.split)
        for name, (ab, sq, ys, n) in acc.items():
            r = sums_to_metrics(ab.sum(), sq.sum(), ys.sum(), n.sum())
            for col, label in ((self.client_geo, "by_client_geo"), (self.client_iid, "by_client_iid")):
                r[label] = {c: sums_to_metrics(ab[col == c].sum(), sq[col == c].sum(), ys[col == c].sum(), n[col == c].sum())
                            for c in sorted(set(col))}
            r["per_sensor_wape_pct"] = [round(100 * a / y, 4) if y else None for a, y in zip(ab, ys)]
            out[name] = r
        out["primary"]["by_horizon"] = [sums_to_metrics(*(x[h] for x in hz)) for h in range(12)]
        return out


# ---------------------------------------------------------------- run directory and checkpoints

def _atomic_torch(obj, path):
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Runner:
    def __init__(self, cfg, root=None, out_root=None):
        self.cfg = validate(copy.deepcopy(cfg))
        self.root = Path(root) if root else core.load_config()["_out"]
        dev = self.cfg["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device cuda requested but CUDA is not available")
        self.device = torch.device("cuda" if dev == "cuda" or (dev == "auto" and torch.cuda.is_available()) else "cpu")
        out_root = Path(out_root or self.cfg["out_root"])
        self.out_root = out_root if out_root.is_absolute() else core.SRC / out_root
        self.id = run_id(self.cfg)
        self.dir = self.out_root / self.id
        self.hash = config_hash(self.cfg)

    # ---- bookkeeping
    def _write_run(self):
        self.info["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        core.save_json(self.dir / "run.json", self.info)

    def _event(self, event, **kw):
        line = json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), run_id=self.id, event=event, **kw), default=str)
        self.out_root.mkdir(parents=True, exist_ok=True)
        with open(self.out_root / "registry.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def dry_run(self):
        """Validate configuration, data, partitions and output location without training."""
        streams = arms.streams(self.root, self.cfg["arm"])
        _, splits, dates = pipeline.experiment_splits(core.load_config(), self.cfg["history"])
        n_train = {s.name: len(ThreeYearWindows(self.root, self.cfg["policy"], self.cfg["history"], "train", s.scaler_scope, sensors=s.sensors))
                   for s in streams}
        state = "new"
        if (self.dir / "run.json").is_file():
            state = core.load_json(self.dir / "run.json")["status"]
        m = build_model(self.cfg["model"], self.cfg["preset"], self.cfg["in_channels"], seed=self.cfg["seed"])
        return dict(run_id=self.id, config_sha256=self.hash, run_dir=str(self.dir), existing=state, device=str(self.device),
                    streams={s.name: dict(sensors=len(s.sensors), scaler=s.scaler_scope, train_windows=n_train[s.name]) for s in streams},
                    train_dates=dates["train"], val_dates=dates["val"], parameters=count_parameters(m),
                    rounds_per_epoch=self.rounds_per_epoch() if self.cfg["arm"] in FL_ARMS else None)

    def rounds_per_epoch(self):
        f = self.cfg["fl"]
        return math.ceil(self.cfg["budget"]["windows_per_epoch"] / (4 * f["local_steps"] * self.cfg["batch_size"]))

    # ---- model / optimizer setup
    def _make_opt(self, model):
        c = self.cfg
        if c["optimizer"] == "sgd":
            return torch.optim.SGD(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        cls = torch.optim.AdamW if c["optimizer"] == "adamw" else torch.optim.Adam
        return cls(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    def _setup(self):
        c = self.cfg
        self.train = arms.loaders(self.root, c["arm"], c["history"], "train", c["policy"])      # training split only
        if c["arm"] in FL_ARMS:
            self.fl = FederatedRun(dict(fl_config(c), device=str(self.device)), self.root, self.train)
            self.models = {"global": self.fl.model}
            self.opts = {}
        else:
            self.fl = None
            self.models = {n: build_model(c["model"], c["preset"], c["in_channels"], seed=c["seed"]).to(self.device) for n in sorted(self.train)}
            self.opts = {n: self._make_opt(m) for n, m in self.models.items()}
        sens = arms.streams(self.root, c["arm"])
        self.owner = np.empty(200, object)
        for s in sens:
            self.owner[s.sensors] = s.name

    def predict(self, X, s, models=None):
        models = models or self.models
        for m in models.values():
            m.eval()
        if len(models) == 1:
            return next(iter(models.values()))(X)
        out = torch.empty(X.shape[0], 12, device=X.device)                    # local arm: each sensor's own client model
        own = self.owner[s]
        for name, m in models.items():
            rows = np.flatnonzero(own == name)
            if len(rows):
                out[torch.as_tensor(rows, device=X.device)] = m(X[torch.as_tensor(rows, device=X.device)])
        return out

    def _state(self):
        st = dict(models={n: {k: v.detach().cpu() for k, v in m.state_dict().items()} for n, m in self.models.items()},
                  optimizers={n: o.state_dict() for n, o in self.opts.items()})
        if self.fl:
            st["fl_server"] = self.fl.alg.state_dict()
            st["fl_round"] = self.fl.round
        st["meta_json"] = json.dumps(dict(run_id=self.id, config_sha256=self.hash, epoch=self.epoch, best=self.info["best"],
                                          bad_epochs=self.bad, counters=self.info["counters"]), default=str)
        return st

    def _load_state(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = json.loads(ck["meta_json"])
        if meta["config_sha256"] != self.hash or meta["run_id"] != self.id:
            raise ValueError(f"{path} belongs to a different configuration")
        for n, m in self.models.items():
            m.load_state_dict({k: v.to(self.device) for k, v in ck["models"][n].items()})
        for n, o in self.opts.items():
            o.load_state_dict(ck["optimizers"][n])
        if self.fl:
            self.fl.alg.load_state_dict(ck["fl_server"])
            self.fl.round = ck["fl_round"]
        return meta

    # ---- training
    def _epoch_central_local(self, epoch):
        c = self.cfg
        names = sorted(self.train)
        W = c["budget"]["windows_per_epoch"] // len(names)
        stats = dict(steps=0, skipped=0, windows=0, valid_targets=0, loss_sum=0.0)
        for n in names:
            i = zlib.crc32(n.encode())                               # stable per-client seed, independent of the other clients
            ld, model, opt = self.train[n], self.models[n], self.opts[n]
            rng = np.random.default_rng(_seed(c["seed"], 10, i, epoch))
            idx = rng.permutation(arms.epoch_sample(len(ld), W, _seed(c["seed"], 11, i), epoch))
            torch.manual_seed(_seed(c["seed"], 12, i, epoch))
            for j in range(0, len(idx), c["batch_size"]):
                b = batch_tensors(ld, idx[j:j + c["batch_size"]], self.device, c["in_channels"])
                loss = train_step(model, opt, b, c["loss"], c["grad_clip"])
                stats["windows"] += len(b[0])
                if loss is None:
                    stats["skipped"] += 1
                    continue
                stats["steps"] += 1; stats["valid_targets"] += int(b[2].sum()); stats["loss_sum"] += loss
        return stats

    def _epoch_fl(self, epoch):
        stats = dict(steps=0, skipped=0, windows=0, valid_targets=0, loss_sum=0.0, rounds=0, upload_bytes=0, download_bytes=0)
        for _ in range(self.rounds_per_epoch()):
            a = self.fl.run_round()
            stats["rounds"] += 1
            stats["upload_bytes"] += a["upload_bytes"]; stats["download_bytes"] += a["download_bytes"]
            for cl in a["clients"].values():
                stats["steps"] += cl["steps"]; stats["skipped"] += cl["skipped_batches"]; stats["valid_targets"] += cl["valid_targets"]
                stats["windows"] += (cl["steps"] + cl["skipped_batches"]) * self.cfg["batch_size"]
                stats["loss_sum"] += (cl["loss"] or 0.0) * cl["steps"]
        self.fl.log = []                                               # accounting is summed into run.json; keep memory bounded
        return stats

    def _flops_per_window(self):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            m = copy.deepcopy(next(iter(self.models.values()))).cpu().eval()
            with FlopCounterMode(display=False) as fc:
                m(torch.zeros(8, 12, self.cfg["in_channels"]))
            return fc.get_total_flops() / 8
        except Exception:
            return None

    def run(self, resume=False, stop_after_epochs=None):
        c = self.cfg
        exists = (self.dir / "run.json").is_file()
        if exists:
            info = core.load_json(self.dir / "run.json")
            if info["config_sha256"] != self.hash:
                raise ValueError(f"{self.dir} holds a different configuration; refusing to reuse it")
            if info["status"] in STATUS_DONE:
                print(f"{self.id}: already {info['status']}; skipped")
                return info
            if not resume:
                raise FileExistsError(f"{self.dir} exists with status {info['status']}; pass --resume to continue it")
        elif resume:
            raise FileNotFoundError(f"nothing to resume at {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._setup()
        self.epoch, self.bad = 0, 0
        if exists:
            self.info = info
            meta = self._load_state(self.dir / "latest.pt") if (self.dir / "latest.pt").is_file() else None
            if meta:
                self.epoch, self.bad = meta["epoch"], meta["bad_epochs"]
                self.info["best"], self.info["counters"] = meta["best"], meta["counters"]
                self.info["history"] = [h for h in self.info["history"] if h["epoch"] <= self.epoch]
            self.info.setdefault("resumes", []).append(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), from_epoch=self.epoch, device=str(self.device)))
            self._event("resume", from_epoch=self.epoch)
        else:
            core.save_json(self.dir / "config.json", c)
            self.info = dict(run_id=self.id, status="running", config=c, config_sha256=self.hash, code_sha256=code_hash(),
                             data_manifest_sha256=core.sha256_file(self.root / "manifests" / "dgx_transfer_manifest.json"),
                             env=environment(self.device), started=time.strftime("%Y-%m-%dT%H:%M:%S"), ended=None,
                             parameters=count_parameters(next(iter(self.models.values()))), models=len(self.models),
                             flops_per_window_forward=self._flops_per_window(), best=None, history=[], resumes=[],
                             counters=dict(steps=0, skipped=0, windows=0, valid_targets=0, rounds=0, upload_bytes=0, download_bytes=0,
                                           train_seconds=0.0, val_seconds=0.0),
                             budget_note="windows_per_epoch sampled training windows per epoch summed over all models/clients")
            self._event("start")
        self.info["status"] = "running"
        self.info["env"]["last_device"] = str(self.device)
        self._write_run()
        ev = Evaluator(self.root, c, c["validation"]["max_windows"])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        session = 0
        try:
            while self.epoch < c["budget"]["epochs"]:
                e = self.epoch + 1
                t0 = time.perf_counter()
                st = self._epoch_fl(e) if self.fl else self._epoch_central_local(e)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                tt = time.perf_counter() - t0
                for k in ("steps", "skipped", "windows", "valid_targets", "rounds", "upload_bytes", "download_bytes"):
                    self.info["counters"][k] += st.get(k, 0)
                self.info["counters"]["train_seconds"] += tt
                rec = dict(epoch=e, train_seconds=round(tt, 2), train_loss=st["loss_sum"] / max(st["steps"], 1),
                           **{k: st[k] for k in st if k != "loss_sum"})
                self.epoch = e
                if e % c["validation"]["every_epochs"] == 0 or e == c["budget"]["epochs"]:
                    v = ev.evaluate(lambda X, s: self.predict(X, s), self.device)
                    self.info["counters"]["val_seconds"] += v["seconds"]
                    rec["val"] = {k: v["primary"][k] for k in ("n_valid", "mae", "rmse", "wape_pct", "wape_denominator")}
                    rec["val_windows"], rec["val_seconds"] = v["windows"], v["seconds"]
                    if "sensitivity" in v:
                        rec["val_sensitivity_wape_pct"] = v["sensitivity"]["wape_pct"]
                    best = self.info["best"]
                    if best is None or rec["val"]["wape_pct"] < best["val"]["wape_pct"] - c["early_stopping"]["min_delta"]:
                        self.bad = 0
                        self.info["best"] = dict(epoch=e, val=rec["val"], by_client_geo={k: x["wape_pct"] for k, x in v["primary"]["by_client_geo"].items()},
                                                 checkpoint="best.pt")
                        _atomic_torch(self._state(), self.dir / "best.pt")
                    else:
                        self.bad += 1
                self.info["history"].append(rec)
                with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                _atomic_torch(self._state(), self.dir / "latest.pt")
                if self.device.type == "cuda":
                    self.info["peak_gpu_mb"] = round(max(self.info.get("peak_gpu_mb") or 0, torch.cuda.max_memory_allocated() / 2**20), 1)
                self._write_run()
                session += 1
                if self.bad >= c["early_stopping"]["patience"]:
                    self.info["status"] = "early_stopped"
                    break
                if stop_after_epochs and session >= stop_after_epochs and self.epoch < c["budget"]["epochs"]:
                    self.info["status"] = "interrupted"
                    self._write_run(); self._event("paused", epoch=self.epoch)
                    return self.info
            if self.info["status"] == "running":
                self.info["status"] = "completed"
            if c["validation"]["final_full"] and c["validation"]["max_windows"] is not None and (self.dir / "best.pt").is_file():
                self._load_state(self.dir / "best.pt")
                full = Evaluator(self.root, c, None).evaluate(lambda X, s: self.predict(X, s), self.device)
                self.info["best"]["val_full"] = {k: full["primary"][k] for k in ("n_valid", "mae", "rmse", "wape_pct", "wape_denominator")}
                self.info["best"]["val_full_by_client_geo"] = {k: x["wape_pct"] for k, x in full["primary"]["by_client_geo"].items()}
            self.info["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._write_run(); self._event(self.info["status"], epoch=self.epoch)
            return self.info
        except BaseException as ex:
            self.info["status"] = "interrupted" if isinstance(ex, KeyboardInterrupt) else "failed"
            self.info["failure"] = "".join(traceback.format_exception_only(type(ex), ex)).strip()
            self._write_run(); self._event(self.info["status"], error=self.info["failure"])
            raise


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON run configuration (merged over aiori2/configs/runs/defaults.json)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. --set arm=local --set budget.epochs=2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-epochs", type=int, help="pause (status interrupted) after N epochs in this session")
    ap.add_argument("--out-root")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, _parse_set(a.set))
    r = Runner(cfg, out_root=a.out_root)
    if a.dry_run:
        print(json.dumps(r.dry_run(), indent=1, default=str))
        return
    info = r.run(resume=a.resume, stop_after_epochs=a.stop_after_epochs)
    print(json.dumps(dict(run_id=info["run_id"], status=info["status"], epochs=len(info["history"]), best=info["best"],
                          counters=info["counters"], dir=str(r.dir)), indent=1, default=str))


if __name__ == "__main__":
    main()
