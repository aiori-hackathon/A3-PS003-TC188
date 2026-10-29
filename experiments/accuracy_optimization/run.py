"""Run one or more accuracy-optimisation experiments from a JSON list of configs (training and validation only).

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/run.py configs/stage1.json [--only ID ...]

Each config is a dict of aopt.train.Config fields. Completed experiment ids already in the registry are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from aopt import data as D  # noqa: E402
from aopt.train import OUT, Config, run  # noqa: E402


def done_ids():
    reg = OUT / "registry.jsonl"
    if not reg.exists():
        return set()
    return {json.loads(line)["exp_id"] for line in reg.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    path = Path(args.config)
    cfgs = json.loads((path if path.is_absolute() or path.exists() else HERE / path).read_text())
    store = D.load(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for c in cfgs:
        if args.only and c["exp_id"] not in args.only:
            continue
        if c["exp_id"] in done_ids():
            print(f"skip {c['exp_id']} (done)", flush=True)
            continue
        try:
            run(Config(**c), store)
        except Exception as e:  # noqa: BLE001 - record and continue with the next experiment
            import traceback
            traceback.print_exc()
            with open(OUT / "registry.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(exp_id=c["exp_id"], status="failed", error=repr(e), config=c)) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
