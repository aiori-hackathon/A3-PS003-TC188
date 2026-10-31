# Accuracy optimisation report (2026-10-29)

**Branch:** `accuracy-optimization`. **Protocol:** `docs/accuracy_optimization_protocol.md`, committed at 07:11 UTC before any stage-2 result.
**Machine-readable ranking:** `docs/accuracy_optimization_results.json`. All private artifacts are under `results/accuracy_optimization/` (git-ignored).

## Summary
- **Best candidate:** a 1-day-context residual MLP, trained local-only on each client.
- **Independent Oct–Dec 2023 result:** WAPE **7.669%**, against **8.834%** for the frozen Phase 9 model on the same windows. That is 1.17 points, or 13.2% relative, better, with all four clients and all three months improving.
- **Validation WAPE:** **6.794%**, against 8.223% for the reproduced baseline. With an out-of-fold LightGBM residual correction it reaches 6.566% on validation. That gain did **not** carry over to Oct–Dec (7.667%).
- **The 6% target was not reached** on validation or on independent data.
- **Noise-floor diagnosis:** 6% sits close to what 5-minute counts allow.
  - Poisson count noise alone gives about 4.96% WAPE.
  - A non-causal oracle that sees the neighbouring intervals t±1 and t±2 gives 5.81%.
  - The 12-step average includes horizons up to 60 minutes.

## A. Baseline
- **Reproduced validation metrics:** the frozen local_seed1 model, clipped at 0, on the Phase 9 validation windows (Jul 11 – Aug 7):
  - WAPE 8.223%, MAE 16.451, RMSE 27.431, sMAPE 16.136%.
  - The GPU pipeline reproduces its frozen predictions to within 0.007 vehicles; the frozen MAE was 16.460 raw and 16.451 clipped.
- **Frozen held-out test result:** **8.672%** WAPE (Aug 8 – Sep 30). It stays the frozen model's result and was not used here.
- **Diagnosed error sources:**
  - **Horizon:** WAPE grows from 6.38% at 5 minutes to 9.85% at 60 minutes.
  - **Clients:** Inland/Central has the highest error (10.57%).
  - **Context:** 12 steps is too short to see the daily pattern.
  - **Loss:** an MSE loss on z-scores weights low-variance sensors heavily, which does not match WAPE.
  - **Low volume:** high relative error at low flow (25% WAPE below 51 vehicles), but those cells carry only 8% of absolute error.
  - **Bias:** high-volume cells are under-predicted (−13.7 vehicles above 488).
  - **Missing inputs:** a negligible share.

## B. Experiments, ranked by validation WAPE
- **Common settings:** single seed, validation split.
- **Budget:** screening runs (s1) had 15,000 steps per client, and some hit the cap while still improving. s2 runs had 40,000 steps with cosine learning-rate decay.

| # | Experiment | Algorithm | Arch | Inputs | Loss | Val WAPE | Val MAE | Val RMSE | Val sMAPE | vs baseline | Min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MLP + residual (OOF) | ensemble+residual | combination | as #2 | - | 6.566%* | 13.14 | 23.39 | 13.16% | +20.1% | - |
| 2 | s2_mlp512_L288_emb | local | MLP | L288, calendar, profile, lags 1d/1w, sensor emb | L1 raw | **6.794%** | 13.593 | 23.776 | 13.785% | +17.4% | 43.7 |
| 3 | s1_l1raw_all_L72 | local | GRU64 | L72, calendar, profile, lags | L1 raw | 7.269% | 14.544 | 24.705 | 14.693% | +11.6% | 6.6 |
| 4 | s1_central_all_L72_emb | centralised | GRU64 | L72, all, sensor emb | L1 raw | 7.318% | 14.642 | 24.840 | 14.908% | +11.0% | 2.5 |
| 5 | s1_mlp_all_L72 | local | MLP256 | L72, all | L1 raw | 7.344% | 14.693 | 24.810 | 14.912% | +10.7% | 2.7 |
| 6 | fl_fedavg | FedAvg | GRU64 | L72, all | L1 raw | 7.458% | 14.921 | 25.088 | 14.935% | +9.3% | 35.3 |
| 7 | s1_l1raw_all | local | GRU64 | L12, all | L1 raw | 7.497% | 14.999 | 25.264 | 14.982% | +8.8% | 2.4 |
| 8 | s1_l1raw_L72 | local | GRU64 | L72 | L1 raw | 7.593% | 15.192 | 25.918 | 15.337% | +7.7% | 6.3 |
| 9 | s1_l1raw_prof | local | GRU64 | L12, calendar, profile | L1 raw | 7.728% | 15.462 | 25.940 | 15.379% | +6.0% | 2.9 |
| 10 | s1_l1raw_L36 | local | GRU64 | L36 | L1 raw | 7.741% | 15.488 | 26.404 | 15.558% | +5.9% | 3.8 |
| 11 | s1_l1raw | local | GRU64 | L12 | L1 raw | 8.153% | 16.312 | 27.454 | 16.341% | +0.8% | 2.7 |
| 12 | s1_base_mse | local | GRU64 | L12 (frozen recipe) | MSE z | 8.405% | 16.817 | 27.869 | 16.931% | −2.2% | 2.5 |

\* Recomputed in `finalize_candidate.py` with missing past readings left as NaN. A first run in `combine.py`, which zero-filled them, gave 6.569%.

**Completed, comparable FL runs: FedAvg only.** At a budget matched to #3, it scored 7.458% against 7.269% for local-only training, so plain averaging is hurt by client heterogeneity, as in Phase 9.

**Implemented and tested but stopped before completion** (deadline; no comparable result exists, so none is claimed):
- **FedPer** (shared GRU body, local heads): 7.81% at round 10 of 30.
- **FedAvg + local fine-tuning, Ditto, FedAdam:** queued, not started.
- **GRU128, 144-step context, sensor embeddings:** 2 of 4 clients finished. LA 5.70% and Bay Area 6.96%, slightly better than the MLP on those clients, at about 3× the cost.
- **MLP with a 2-day context:** LA at 6.01% at step 18,000 when stopped.
- **MLP with a 1-week context:** not started.

**Dropped by evidence:**
- **FedProx (small μ):** behaves like FedAvg.
- **FedYogi and FedAvgM:** server-optimiser variants of the FedAdam family; FedAvg already trailed local-only training.

**Not attempted:**
- **SCAFFOLD, FedNova, FedDyn:** these address client drift, but sharing itself was not the limiting factor here.
- **Clustering (IFCA):** not meaningful with four clients.
- **FedBN:** the models have no batch normalisation.

## C. Best candidate (candidate v1)
- **Model:** a residual MLP over the last 288 five-minute steps.
  - Each step has scaled causally filled flow, an observed flag and time of day.
  - It adds target-step covariates: time of day, weekday, the training-period weekday × slot median, and the same time yesterday and last week, each with an observed flag.
  - A 16-dimensional sensor embedding feeds the head; there are 12 outputs; negative values are clipped at 0.
  - Shape: 3 blocks × 512 units, GELU, LayerNorm, dropout 0.1.
- **Training:** local-only, one model per client. L1 loss on raw-unit errors (what WAPE sums), Adam at 2e-3 with cosine decay, batch 2048, up to 40,000 steps, early stopping on validation WAPE.
- **Validation (Jul 11 – Aug 7):** WAPE 6.794%, MAE 13.593, RMSE 23.776, sMAPE 13.785%.

  | Horizon | 5 min | 20 min | 35 min | 50 min | 60 min |
  |---|---|---|---|---|---|
  | WAPE | 5.96% | 6.59% | 6.92% | 7.16% | 7.35% |

- **Clients, validation WAPE:** LA 5.80%, Bay Area 7.05%, Coastal 6.32%, Inland 8.76%.
- **Independent Oct–Dec 2023:** 7.669% overall.
  - Clients: LA 6.28%, Bay Area 7.81%, Coastal 8.08%, Inland 9.41%.
  - Months: October 7.30%, November 7.67%, December 8.07%.
  - The frozen model on the same windows: 8.834% overall; clients 7.36 / 8.96 / 9.30 / 10.67%.
- **Residual correction** (LightGBM on past-only features: base forecast, horizon, last reading, mean of last 12, 3-step trend, slot, weekday, sensor):
  - On validation, 2-fold blocked out-of-fold: 6.566%, with every client improving.
  - On Oct–Dec: 7.667%, a negligible gain. Its validation gain came from per-sensor recalibration to the most recent weeks, which does not last for months.
- **Artifacts:** `results/accuracy_optimization/candidate_v1/` holds the MLP weights, config, training profile, residual model and manifest.
- **Inference:**
  - **Input:** one week plus 12 steps of 5-minute history per sensor.
  - **Latency:** 4.4 ms per request on CPU, including the API.
  - **Hardware:** a GPU is not needed for inference.

## D. Target assessment
- **Did any candidate reach below 6% validation WAPE?** No. The best is 6.566% (out-of-fold, with residual correction); the base model is 6.794%.
- **Was the result reproducible?** The checkpoint reproduces its predictions exactly, and the out-of-fold score was recomputed independently. Only one seed per configuration was run; the 3-seed robustness step in the protocol was cut by the deadline.
- **Independent confirmation below 6%?** No. On untouched Oct–Dec 2023 the best is **7.669%**, a verified 13.2% relative improvement over the frozen model's 8.834%.
- **Best verified result:** 7.669% WAPE on independent data, and 6.794% on validation for the base model.
- **What would most likely help:**
  1. Finish the GRU128/144 and long-context (2-day, 1-week) runs; partial results were already better on two clients.
  2. Run 3 seeds and average them.
  3. Retrain on training plus validation data before deployment.
  4. Regular retraining or online recalibration, since the error rises from October to December.
  5. Spatial information from neighbouring sensors, for example a graph model, needing confirmed geography.
  6. Coarser aggregation, such as 15-minute intervals: WAPE below 6% at 5 minutes is near the count-noise floor.

## E. Engineering handoff
**Environments:**
- `.venv`: the unchanged backend environment.
- `.venv-gpu`: torch 2.14.1+cu130, numpy, pandas, scikit-learn, LightGBM and Optuna (Optuna unused); git-ignored.
- **GPU:** RTX PRO 4000 Blackwell, verified with a real matmul. Up to 96% utilisation with three parallel queues.

**Commands:**
```powershell
.venv\Scripts\python.exe experiments\accuracy_optimization\diagnose_baseline.py
.venv-gpu\Scripts\python.exe experiments\accuracy_optimization\verify_pipeline.py
.venv-gpu\Scripts\python.exe experiments\accuracy_optimization\run.py configs\stage1.json
.venv-gpu\Scripts\python.exe experiments\accuracy_optimization\finalize_candidate.py
.venv-gpu\Scripts\python.exe experiments\accuracy_optimization\make_report.py
.venv-gpu\Scripts\python.exe -m pytest experiments\accuracy_optimization\test_aopt.py
```
The other configs are `configs\stage2*.json` and `stage3A.json`. The evaluation script `eval_untouched.py` has already run once and refuses to run again.

**Backend:**
- `app/backend/candidate.py` adds model key `candidate` **only** when `AIORI_CANDIDATE=1`, for historical requests only; a request with 12 user readings gets a clear 400.
- **Rollback:** unset the variable. The original models, the frozen results and the React front end are unchanged; the front end does not show the candidate.
- **Tests:** `app/tests/test_candidate.py` checks that CPU inference equals the validated predictions.

**Tests passing:**
- backend: 23;
- accuracy-pipeline leakage and metric tests: 4;
- the frozen-model reproduction check.

**Deployment recommendation:**
- Keep the frozen model as the default and the reference for its 8.672% held-out result.
- Offer candidate v1 as an experimental option for historical replay.
- Before promoting it:
  1. finish the seed-robustness runs;
  2. decide whether to retrain on training plus validation data;
  3. extend the future-forecast interface to accept a week of readings.
- Do not serve the residual correction, which gave no independent benefit.
