"""Experimental candidate model (opt-in): AIORI_CANDIDATE=1 adds model key "candidate" to POST /predict.

Candidate v1 = the validation-selected 1-day-context residual MLP from experiments/accuracy_optimization, one model per
client, clipped at 0. It needs one week of 5-minute history per sensor (2016 steps for same-time-last-week features),
so it serves historical requests only; forecasts from 12 user readings still use the original models.
The residual LightGBM correction is not served: on the independent Oct-Dec 2023 evaluation it changed WAPE by
0.002 points, so it adds a dependency without a measurable benefit.

Validation WAPE (Jul 11 - Aug 7, 2023) 6.794%; independent Oct-Dec 2023 WAPE 7.669% (original final model 8.834% on the
same windows). The original model and its frozen 8.672% held-out test result are unchanged; remove the variable to roll back.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(os.environ.get("AIORI_CANDIDATE_DIR", ROOT / "results" / "accuracy_optimization" / "candidate_v1"))
sys.path.insert(0, str(ROOT / "experiments" / "accuracy_optimization"))

from aopt import data as D  # noqa: E402
from aopt.train import Config, make_model  # noqa: E402

LABEL = "Optimized MLP (1-day context, not federated)"


class Candidate:
    def __init__(self, run, scaler_mean, scaler_std, device):
        cfg = Config(**json.loads((DIR / "mlp_config.json").read_text(encoding="utf-8")))
        self.cfg, self.f, self.device = cfg, cfg.features(), device
        T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)  # noqa: E731
        client = np.array([D.CLIENTS.index(c) for c in run.client])
        self.store = D.Store(z=T(run.inputs_scaled), obs=T(~np.asarray(run.input_imputed)), flow=T(run.flow_raw),
                             prof=T(np.load(DIR / "profile_scaled.npy")), mean=T(scaler_mean), std=T(scaler_std),
                             client=T(client, torch.long), train=None, val=None, val_cells=0.0, device=device, n_sensors=len(client))
        states = torch.load(DIR / "mlp_models.pt", map_location=device, weights_only=True)
        self.models = {}
        for k, sd in states.items():
            m = make_model(cfg, self.store)
            m.load_state_dict(sd)
            self.models[int(k)] = m.eval()
        self.client = client
        self.manifest = json.loads((DIR / "manifest.json").read_text(encoding="utf-8"))

    @torch.no_grad()
    def forecast(self, pos, t0):
        if t0 < D.WEEK + self.cfg.input_len:
            raise ValueError("needs one week of history")
        b = D.batch(self.store, torch.tensor([[pos, t0]], device=self.device), self.f)
        z = self.models[int(self.client[pos])](b["x"], b["xf"], b["s"])
        raw = z * self.store.std[pos] + self.store.mean[pos]
        return raw.clamp(min=0)[0].cpu().numpy().astype(float)

    def provenance(self):
        oof = self.manifest["validation_with_residual_oof"]
        base = self.manifest["validation_base_alone"]
        return dict(arm="Local-only residual MLP (experimental)", seed=0, architecture=self.manifest["base_model"]["architecture"],
                    selection="lowest validation WAPE among the accuracy-optimisation runs (docs/accuracy_optimization_protocol.md)",
                    routing="each sensor uses its own client's model", negative_clipping=True,
                    validation_wape_pct=round(base["wape_pct"], 3), validation_mae=base["mae"],
                    independent_oct_dec_2023_wape_pct=7.669, residual_corrected_validation_wape_pct_oof=round(oof["wape_pct"], 3),
                    note="experimental; historical requests only; not the frozen final model", units="vehicles per 5-minute interval")
