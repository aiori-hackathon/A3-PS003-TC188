"""Bounded throughput benchmark on real data (warm-up, CUDA-synchronised timing, repeated measurements).

    python -m aiori2.bench [--device cpu|cuda] [--batch 64] [--out FILE]

Measures, per model: centralised training windows/s (data loading + step), data-loading alone, FedAvg round
throughput (4 clients, 50 local steps), validation windows/s, and for one model FedAdam/SCAFFOLD rounds and checkpoint
save/load. The JSON it writes feeds `python -m aiori2.plan estimate`. Nothing here trains a usable model.
"""

import argparse
import copy
import io
import statistics
import time

import numpy as np
import torch

from aiori2 import arms, core
from aiori2.fl import FederatedRun, make_config
from aiori2.losses import batch_tensors, train_step
from aiori2.models import MODELS, build_model
from aiori2.run import Evaluator, load_config


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, reps=3):
    out = []
    for _ in range(reps):
        _sync(dev); t = time.perf_counter(); fn(); _sync(dev); out.append(time.perf_counter() - t)
    return statistics.median(out), out


def bench(device, batch=64, steps=200, local_steps=50, val_windows=200_000):
    dev = torch.device(device)
    root = core.load_config()["_out"]
    rng = np.random.default_rng(0)
    central = arms.loaders(root, "centralised", "B", "train")["all"]
    fl_loaders = arms.loaders(root, "fedavg_geo", "B", "train")
    res = dict(device=device, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, torch=torch.__version__, batch=batch,
               steps_per_rep=steps, local_steps=local_steps, train_windows_per_s={}, val_windows_per_s={}, detail={})
    for name in MODELS:
        d = res["detail"][name] = {}
        model = build_model(name, seed=0).to(dev)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        idx = [rng.choice(len(central), batch, replace=False) for _ in range(steps)]
        for i in idx[:20]:
            train_step(model, opt, batch_tensors(central, i, dev))                    # warm-up
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_train, reps = timed(lambda: [train_step(model, opt, batch_tensors(central, i, dev)) for i in idx], dev)
        t_load, _ = timed(lambda: [batch_tensors(central, i, dev) for i in idx], dev)
        d["central_step_ms"] = round(1000 * t_train / steps, 3)
        d["data_loading_share"] = round(t_load / t_train, 3)
        d["central_reps_s"] = [round(x, 3) for x in reps]
        cfg = make_config(algorithm="fedavg", arm="fedavg_geo", history="B", model=name, local_steps=local_steps, batch_size=batch,
                          device=device)
        run = FederatedRun(cfg, root, fl_loaders)
        run.run_round()                                                               # warm-up round
        t_fl, _ = timed(lambda: run.run_round(), dev, reps=2)
        d["fedavg_round_s"] = round(t_fl, 3)
        res["train_windows_per_s"][name] = dict(centralised=round(steps * batch / t_train), fl=round(4 * local_steps * batch / t_fl))
        ev = Evaluator(root, load_config(None, dict(arm="centralised", history="B", model=name, validation=dict(batch_size=8192))), val_windows)
        ev.evaluate(lambda X, s: model(X), dev)                                      # warm-up
        t_val, _ = timed(lambda: ev.evaluate(lambda X, s: model(X), dev), dev, reps=2)
        res["val_windows_per_s"][name] = round(val_windows / t_val)
        if dev.type == "cuda":
            d["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        print(name, res["train_windows_per_s"][name], "val", res["val_windows_per_s"][name], d, flush=True)
    name = "mlp"
    for alg in ("fedadam", "scaffold"):
        cfg = make_config(algorithm=alg, arm="fedavg_geo", history="B", model=name, local_steps=local_steps, batch_size=batch, device=device,
                          server_lr=0.01 if alg == "fedadam" else 1.0)
        run = FederatedRun(cfg, root, fl_loaders)
        run.run_round()
        t, _ = timed(lambda: run.run_round(), dev, reps=2)
        res["detail"][f"{name}_{alg}_round_s"] = round(t, 3)
        res["detail"][f"{name}_{alg}_windows_per_s"] = round(4 * local_steps * batch / t)
    m = build_model("patchtst", seed=0)
    buf = io.BytesIO()
    t_save, _ = timed(lambda: (buf.seek(0), torch.save(copy.deepcopy(m.state_dict()), buf)), torch.device("cpu"))
    t_load, _ = timed(lambda: (buf.seek(0), torch.load(buf, weights_only=True)), torch.device("cpu"))
    res["detail"]["checkpoint_save_ms"], res["detail"]["checkpoint_load_ms"] = round(1000 * t_save, 2), round(1000 * t_load, 2)
    res["val_windows_total"] = len(Evaluator(root, load_config(None, dict(arm="centralised", history="B")), None).positions)
    return res


def main():
    ap = argparse.ArgumentParser(prog="python -m aiori2.bench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = bench(a.device, a.batch)
    out = a.out or core.load_config()["_out"] / "phase6" / f"bench_{a.device}_b{a.batch}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    core.save_json(out, res)
    print(out)


if __name__ == "__main__":
    main()
