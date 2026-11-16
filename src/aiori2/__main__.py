"""CLI: python -m aiori2 [--config FILE] [--base-dir SRC] [--out DIR] STEP [STEP ...]   (run from the src/ directory)

Steps: protect extract assemble quality experiments manifest verify, or `all` for the whole pipeline in that order.
"""

import argparse
import json
import logging
import shutil
import sys
import time
import tracemalloc

from aiori2 import core, pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aiori2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=list(pipeline.STEPS) + ["all"])
    ap.add_argument("--config"); ap.add_argument("--base-dir"); ap.add_argument("--out")
    ap.add_argument("--min-free-gb", type=float, default=20.0, help="refuse to write when less disk space is free")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = core.load_config(a.config, a.base_dir, a.out)
    cfg["_out"].mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg["_out"]).free / 1e9
    if free < a.min_free_gb:
        sys.exit(f"STOP: only {free:.0f} GB free at {cfg['_out']}")
    steps = list(pipeline.STEPS) if a.steps == ["all"] else a.steps
    runlog = core.load_json(cfg["_out"] / "manifests" / "run_log.json", [])
    tracemalloc.start()
    for s in steps:
        tracemalloc.reset_peak()
        t0 = time.time()
        res = pipeline.STEPS[s](cfg)
        entry = dict(step=s, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.time() - t0, 1),
                     config_sha256=cfg["_config_sha256"], result=res if s != "protect" else dict(files=len(res.get("files", {}))))
        runlog.append(entry)
        core.save_json(cfg["_out"] / "manifests" / "run_log.json", runlog)
        print(json.dumps(entry, default=str)[:2000], flush=True)
        if s == "verify" and not res["ok"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
