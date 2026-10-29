"""Check the GPU pipeline against the frozen study: the deployed local_seed1 weights, loaded read-only into
aopt.models.Forecaster, must reproduce the frozen validation predictions (and so the 8.223% validation WAPE).

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/verify_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aopt import data as D  # noqa: E402
from aopt.train import Config, make_model, predict_raw, evaluate  # noqa: E402

ROOT = HERE.parents[1]
RUN = ROOT / "results" / "phase9" / "runs" / "local_seed1"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False      # full fp32, as in the frozen CPU study
    torch.backends.cuda.matmul.allow_tf32 = False
    store = D.load(dev)
    cfg = Config(exp_id="verify", algorithm="local")
    f = cfg.features()
    models = {}
    for c, name in enumerate(D.CLIENTS):
        sd = torch.load(RUN / f"best_model_{name}.pt", map_location=dev, weights_only=True)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
        m = make_model(cfg, store)
        m.load_state_dict({("body." + k[4:]) if k.startswith("gru.") else ("head.net." + k[5:]): v for k, v in sd.items()})
        models[c] = m
    p = predict_raw(models, store, f, store.val).cpu().numpy()
    with np.load(RUN / "predictions_val.npz") as z:
        ref = np.maximum(z["y_hat"], 0)
        order = np.lexsort((z["t0"], z["sensor_pos"]))
    v = store.val.cpu().numpy()
    mine = np.lexsort((v[:, 1], v[:, 0]))
    diff = np.abs(p[mine] - ref[order]).max()
    rep = evaluate(models, store, f)
    print(f"max |difference| vs frozen predictions: {diff:.5f} vehicles; validation WAPE {rep['wape_pct']:.4f}% MAE {rep['mae']:.4f}")
    assert diff < 0.05, "pipeline does not reproduce the frozen model"


if __name__ == "__main__":
    main()
