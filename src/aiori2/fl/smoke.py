"""Bounded smoke test: 8 algorithms x 5 models on real training batches, 2 rounds, 2 local steps, batch 16.

    python -m aiori2.fl.smoke [--device cpu|cuda]

IID and Non-IID arms and histories A/B alternate across the 40 combinations. Writes
<data>/phase5/fl_smoke_<device>_<time>.json. No checkpoints, no evaluation, no training beyond these steps.
"""

import argparse
import itertools
import time

import torch

from aiori2 import arms, core
from aiori2.fl import ALGORITHMS, FederatedRun, make_config
from aiori2.models import MODELS


def run(device):
    root = core.load_config()["_out"]
    cache = {}
    out = dict(device=device, torch=torch.__version__, rounds=2, local_steps=2, batch_size=16, runs=[])
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    for i, (alg, model) in enumerate(itertools.product(ALGORITHMS, MODELS)):
        arm, hist = ("fedavg_iid", "fedavg_geo")[i % 2], ("A", "B")[(i // 2) % 2]
        key = (arm, hist)
        if key not in cache:
            cache[key] = arms.loaders(root, arm, hist, "train")
        cfg = make_config(algorithm=alg, model=model, arm=arm, history=hist, local_steps=2, batch_size=16, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = FederatedRun(cfg, root, cache[key])
        log = r.run(2)
        if device == "cuda":
            torch.cuda.synchronize()
        ok = all(bool(torch.isfinite(v).all()) for v in r.model.state_dict().values())
        out["runs"].append(dict(algorithm=alg, model=model, arm=arm, history=hist, finite=ok, seconds=round(time.perf_counter() - t0, 3),
                                peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
                                aggregated=[x["aggregated"] for x in log], steps=[[c["steps"] for c in x["clients"].values()] for x in log],
                                bytes_per_round=log[-1]["total_bytes"], loss=[round(c["loss"], 5) for c in log[-1]["clients"].values()]))
        print(f"{alg:10s} {model:8s} {arm:10s} {hist} finite={ok} {out['runs'][-1]['seconds']}s", flush=True)
    out["all_finite"] = all(x["finite"] for x in out["runs"])
    path = root / "phase5" / f"fl_smoke_{device}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    core.save_json(path, out)
    print(path)
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m aiori2.fl.smoke")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(ap.parse_args().device)


if __name__ == "__main__":
    main()
