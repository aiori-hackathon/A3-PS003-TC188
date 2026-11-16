"""Bounded smoke test of the five forecasters on real Phase 3 batches (no training run).

    python -m aiori2.models.smoke [--device cpu|cuda] [--preset medium]

For each model: batch sizes 1/8/32/64 from experiment A and B (centralised loader), forward + masked loss +
backward + one Adam step, timing (CUDA-synchronised) and peak GPU memory; one step on a client stream of every arm;
checkpoint save -> reload -> identical predictions. At most 105 windows per history per model.
Writes <data>/phase4/smoke_<device>_<time>.json and checkpoints under results/aiori2/phase4_smoke/<time>/ (new folder).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aiori2 import arms, core
from aiori2.losses import batch_tensors, masked_loss, train_step
from aiori2.models import MODELS, build_model, count_parameters, load_checkpoint, save_checkpoint

SEED = 20261117


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run(device="cpu", preset=None):
    cfg = core.load_config()
    root = cfg["_out"]
    dev = torch.device(device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ck_dir = core.SRC / "results" / "aiori2" / "phase4_smoke" / f"{stamp}_{dev.type}"
    rep = dict(device=str(dev), torch=torch.__version__, cuda_build=torch.version.cuda, preset=preset, models={})
    if dev.type == "cuda":
        rep["gpu"] = torch.cuda.get_device_name(0)
    loaders = {h: arms.loaders(root, "centralised", h, "train")["all"] for h in arms.HISTORIES}
    for name in MODELS:
        torch.manual_seed(SEED)
        model = build_model(name, preset, seed=SEED).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        r = dict(parameters=count_parameters(model), steps=[])
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for h, ld in loaders.items():
            for bs in (1, 8, 32, 64):
                batch = batch_tensors(ld, arms.epoch_sample(len(ld), bs, SEED, bs), dev)
                _sync(dev); t0 = time.perf_counter()
                loss = train_step(model, opt, batch)
                _sync(dev); dt = time.perf_counter() - t0
                grads_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                with torch.no_grad():
                    pred = model.eval()(batch[0])
                r["steps"].append(dict(history=h, batch=bs, out_shape=list(pred.shape), out_dtype=str(pred.dtype),
                                       finite_pred=bool(torch.isfinite(pred).all()), loss=loss, grads_finite=grads_finite,
                                       step_ms=round(dt * 1000, 2)))
        if dev.type == "cuda":
            r["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        # one masked update on a client stream of every arm (B), each with its own scaler
        r["arms"] = {}
        for arm in arms.ARMS:
            st = arms.loaders(root, arm, "B", "train")
            cname, ld = next(iter(st.items()))
            client_model = build_model(name, preset, seed=SEED).to(dev)
            b = batch_tensors(ld, arms.epoch_sample(len(ld), 32, SEED, 0), dev)
            l = train_step(client_model, torch.optim.Adam(client_model.parameters(), 1e-3), b)
            r["arms"][arm] = dict(stream=cname, scaler=ld.scope, loss=l, sensors_in_batch_are_own=bool(np.isin(b[5].cpu().numpy(), ld.sensors).all()))
        # checkpoint round trip
        path = ck_dir / f"{name}.pt"
        save_checkpoint(model.cpu(), path, dict(seed=SEED, purpose="phase4 smoke test only"))
        model.to(dev)
        re, meta = load_checkpoint(path, dev)
        x = batch_tensors(loaders["A"], np.arange(16), dev)[0]
        with torch.no_grad():
            same = bool(torch.equal(model.eval()(x), re.eval()(x)))
        r["checkpoint"] = dict(path=path.relative_to(core.SRC).as_posix(), reload_identical=same, data_contract=meta["data_contract"])
        rep["models"][name] = r
        print(name, r["parameters"], "ok" if same and all(s["finite_pred"] and s["grads_finite"] for s in r["steps"]) else "PROBLEM", flush=True)
    out = root / "phase4" / f"smoke_{dev.type}_{stamp}.json"
    core.save_json(out, rep)
    print(out)
    return rep


def main():
    ap = argparse.ArgumentParser(prog="python -m aiori2.models.smoke")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preset")
    a = ap.parse_args()
    run(a.device, a.preset)


if __name__ == "__main__":
    main()
