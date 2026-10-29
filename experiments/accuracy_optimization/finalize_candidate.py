"""Freeze candidate v1 (validation-selected): 1-day-context MLP (s2_mlp512_L288_emb) + LightGBM residual correction.

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/finalize_candidate.py

1. Re-runs the saved MLP checkpoint on the validation windows and checks it reproduces its saved predictions.
2. Re-computes the residual correction out-of-fold (2-fold blocked inside validation) from scratch: pooled WAPE and
   all four clients, plus the base model alone on the same cells.
3. Fits the deployable correction on the whole validation period (for application to later, unseen windows) and
   saves everything to results/accuracy_optimization/candidate_v1/ (private; git-ignored).
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aopt import data as D, residual as R  # noqa: E402
from aopt.metrics import full_report  # noqa: E402
from aopt.train import Config, make_model, predict_raw  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "results" / "accuracy_optimization"
BASE = "s2_mlp512_L288_emb"
DEST = OUT / "candidate_v1"
SPLIT_T = 55008 + 14 * 288


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = D.load(dev)
    ck = OUT / "checkpoints" / BASE
    cfg = Config(**json.loads((ck / "config.json").read_text()))
    states = torch.load(ck / "models.pt", map_location=dev, weights_only=True)
    models = {}
    for k, sd in states.items():
        m = make_model(cfg, store)
        m.load_state_dict(sd)
        models[int(k)] = m
    pred = predict_raw(models, store, cfg.features(), store.val).double().cpu().numpy()
    saved = np.load(OUT / "validation_predictions" / f"{BASE}.npy")
    repro = float(np.abs(pred - saved).max())
    assert repro < 0.01, f"checkpoint does not reproduce its predictions ({repro})"

    v = store.val.cpu().numpy()
    flow = store.flow.cpu().numpy().astype(np.float64)
    y = flow[v[:, 0][:, None], v[:, 1][:, None] + np.arange(12)]
    m = np.isfinite(y)
    a = np.where(m, y, 0.0)
    client = store.client.cpu().numpy()[v[:, 0]]
    X = R.features(pred, v, flow)
    assert np.isfinite(X[..., [0, 1, 5, 6, 7]]).all()

    foldA = v[:, 1] + 11 < SPLIT_T
    oof = pred.copy()
    for fit_rows, score_rows in ((foldA, ~foldA), (~foldA, foldA)):
        model = R.fit(X[fit_rows], (a - pred)[fit_rows], m[fit_rows])
        oof[score_rows] = R.apply(model, pred[score_rows], X[score_rows])
    rep_base = full_report(np.maximum(pred, 0), a, m, v[:, 0], v[:, 1], client, D.CLIENTS)
    rep_oof = full_report(oof, a, m, v[:, 0], v[:, 1], client, D.CLIENTS)
    pooled = 100 * np.abs(np.where(m, oof - a, 0)).sum() / a.sum()      # direct pooled check
    assert abs(pooled - rep_oof["wape_pct"]) < 1e-9

    final = R.fit(X, a - pred, m)
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy(ck / "models.pt", DEST / "mlp_models.pt")
    shutil.copy(ck / "config.json", DEST / "mlp_config.json")
    final.booster_.save_model(str(DEST / "residual_lgbm.txt"))
    np.save(DEST / "profile_scaled.npy", store.prof.cpu().numpy())
    for r in (rep_base, rep_oof):
        r.pop("by_hour_share_of_abs_error", None)
    manifest = dict(
        name="candidate_v1", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="experimental, validation-selected",
        base_model=dict(exp_id=BASE, architecture="residual MLP (3 blocks x 512, GELU, LayerNorm, dropout 0.1) + 16-d sensor embedding, one model per client",
                        training="local-only (each client trains on its own sensors), L1 loss on raw units, Adam lr 2e-3 cosine, batch 2048, early stopping on validation WAPE",
                        reproduced_max_abs_diff=repro),
        input_contract=dict(history_steps_required=2016 + 12, past_window=288, past_features=["scaled filled flow", "observed flag", "slot sin", "slot cos"],
                            future_covariates=["slot sin/cos", "weekday sin/cos", "training weekday x slot median", "same time yesterday (+flag)", "same time last week (+flag)"],
                            note="needs one week of 5-minute history per sensor; the app's 12-reading future mode cannot supply it"),
        preprocessing="frozen nine-month scaler, causal fill and training-only profile (profile_scaled.npy); see aopt/data.py",
        residual=dict(features=R.FEATURES, params=R.PARAMS, fitted_on="all validation windows (Jul 11 - Aug 7, 2023) for deployment",
                      score="out-of-fold, 2-fold blocked within validation"),
        validation_base_alone=rep_base, validation_with_residual_oof=rep_oof,
        reference=dict(frozen_final_model_validation_wape_pct=8.2226, frozen_final_model_test_wape_pct=8.672,
                       note="8.672% is the frozen Phase 9 held-out test result of the original model, not of this candidate"))
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(f"reproduced base predictions (max diff {repro:.5f}); base val WAPE {rep_base['wape_pct']:.3f}%; "
          f"with residual (OOF) {rep_oof['wape_pct']:.3f}% (pooled check {pooled:.3f}%)")
    for c in D.CLIENTS:
        print(f"  {c}: {rep_base['by_client'][c]['wape_pct']:.3f}% -> {rep_oof['by_client'][c]['wape_pct']:.3f}%")


if __name__ == "__main__":
    main()
