# Phase 8 held-out test evaluation — pre-registered protocol (January 26–31, 2023)

**Date:** 2026-10-28. **Status: DESIGN ONLY — awaiting approval.**

**Not done yet:**
- no test window has been loaded;
- no test prediction or test metric exists;
- the evaluation script below is not written.

**Basis:**
- `docs/phase8_200_sensor_technical_spec.md` (§5);
- `docs/phase8_200_sensor_implementation_plan.md` (§11–§12);
- `docs/phase8_results.md` (the S5 validation results).

---

## 1. Purpose and how test findings are kept apart from validation findings

The January **validation** split (23–25 Jan) was used to choose B1, for early stopping in all 12 runs, and for the S5 analysis. Its results are optimistic for every arm and are worded "on January validation data".

The **test** split (26–31 Jan) has not been read by any model, threshold or analysis step. This protocol scores it **once**, with everything fixed in advance. Its results would be worded "on January test data" and reported **next to** the validation results, never merged with them.

| | Validation (S5, done) | Test (this protocol) |
|---|---|---|
| Period | Mon 23 – Wed 25 Jan (3 weekdays) | Thu 26 – Tue 31 Jan (6 days, **including Sat 28 and Sun 29**) |
| Steps | [6336, 7200) | [7200, 8928) |
| Windows (12 → 12) | 170,600 | 343,246 (stored at S2 and never loaded) |
| Role | B1 choice, early stopping, S5 analysis | one-time confirmation only |
| Wording | "on January validation data" | "on January test data" |

## 2. What is frozen before evaluation

| Item | Frozen value | Where recorded |
|---|---|---|
| Models | The 21 best-checkpoint files of the 12 S4 runs: 3 centralised `best_model.pt`; 12 local-only `best_model_<client>.pt` (4 per seed); 3 IID and 3 geographic `best_model_global.pt` | `results/phase8/test_protocol/frozen_checkpoints.json` (SHA-256 of all 66 run files; manifest SHA-256 `721b2c94db89c73f9f602a890ea6d983507d1839e0e1efd2b29391d3d8d0d505`) |
| Configs | the four `phase8_*_b1.json` files and `phase8_selection.json` (hashes) | same manifest |
| Data | `data/processed/geo200/` (200 sensors; training-only scaler and time-of-day medians; causal input fill; NaN targets with masks) | `checkpoint_manifest.json` artifact hashes, copied into the freeze manifest |
| Code | `aiori/` model, dataset, training, evaluation, metrics, baselines, windows, federated and phase8_analysis (hashes). The new script (§4) is hashed when it is approved, before it runs | same manifest, plus the E1 record |
| SESOI | **0.1638** MAE (1% of the centralised validation MAE), **not recomputed on test** | `results/phase8/analysis/sesoi_log.json` (hash in the manifest) |
| Statistics | the S5 method unchanged (§6) | this document |

**Not allowed after this point:**
- retraining, fine-tuning or recalibration;
- swapping checkpoints or choosing a "better" epoch or round;
- hyperparameter or threshold changes;
- clipping negative predictions;
- changing preprocessing, sensors, clients or partitions.

## 3. Firewall: no test information flows back

1. **Model selection:** every checkpoint was chosen on validation MAE during S4. The test split plays no part in it.
2. **Preprocessing:** the scaler and time-of-day medians were fitted on training steps only (S2, verified). Test inputs use the frozen causal fill.
   - The first test windows take inputs from the last hour of validation. That is past data, allowed by the 12 → 12 window rule.
   - Test targets are never filled; missing targets stay masked.
3. **Thresholds and methods:** the SESOI, bootstrap design, seed, contrasts, Holm family and wording rules are all fixed here, before any test value is seen.
4. **Order:** the script computes everything in one pass and writes the results once. The analysis plan is not revised after the results are seen.
5. **Failure handling:**
   - **Execution defect** (crash, hash mismatch, failed pre-check): stop and report. A rerun needs a documented fix and approval, and the report lists every attempt.
   - **Results nobody likes:** not a reason to rerun.
6. **Sensor-level outputs:** predictions and per-sensor metrics stay under git-ignored `results/phase8/test/`. Only aggregate statistics go into a document.

## 4. Procedure (each step needs approval; this design approves none of them)

| Step | Work | Produces |
|---|---|---|
| **E1** | Write `scripts/phase8_test_eval.py` with **synthetic tests only** (§4.1). Record its SHA-256 | code, tests, hash |
| **E2** | Read-only pre-checks on real data, **validation only**: verify every frozen hash; reload each checkpoint and re-predict validation; require **byte-identical** agreement with the saved S4 validation predictions | pre-check log. Proves the frozen files are exactly the S4 best models |
| **E3** | The single test pass: inference for all 12 runs and the 3 reference forecasts; write prediction records and metrics | `results/phase8/test/` (local) |
| **E4** | Pre-declared analysis (§6) on the E3 records; aggregate report | `results/phase8/test/analysis_test.json` (local); `docs/phase8_test_results.md` (aggregate) |

### 4.1 Planned behaviour of `scripts/phase8_test_eval.py`
**Refuses to run unless:**
- `--protocol-sha256` equals the SHA-256 of this document as approved;
- `--freeze-sha256` equals the manifest hash above;
- every frozen file matches its recorded hash;
- `results/phase8/test/` is absent or empty.

**One-shot guard:** it writes `results/phase8/test/RUN_STARTED` before loading test windows. If that marker exists, it refuses.

**Inference only:**
- `load_run("geo200")` with all three splits;
- `ForecastGRU(4, 64)` with `load_state_dict(strict=True)`, then `model.eval()` and `torch.no_grad()`;
- no optimiser is created;
- each model's parameter hash is checked unchanged after prediction.

**Which model predicts which windows:**

| Arm | Model and windows |
|---|---|
| Centralised | its single model on all test windows |
| Local-only | each client's model predicts **only its own 50 sensors' windows**, as in validation; the four outputs are merged |
| IID and geographic FedAvg | the global model on all test windows |

Per-client results always use the **geographic** regions, including for the IID arm.

**Outputs:**
- per run: `predictions_<arm>_<seed>_test.npz` and `metrics_test.json`, using the existing `evaluation.evaluate_predictions` and `write_prediction_record`;
- `reference_forecasts_test.json` (persistence, daily seasonal naive, historical average) via `evaluation.evaluate_baselines(run, "test")`;
- a run log: hashes, versions, timing, peak memory.

**Checks:**
- predictions finite;
- mask equal to the finite raw targets;
- identical windows, masks and targets across all 12 runs;
- window count equal to 343,246.

**Synthetic tests (E1):**
- the refusals (wrong hashes, an existing marker or non-empty output, the test split never touched during pre-checks);
- the local-only client routing;
- parameters unchanged after prediction;
- no optimiser or `backward` call;
- metric equality with `evaluation.score`.

## 5. Metrics reported (test; validation shown alongside)
- **Primary:** pooled (micro) MAE in raw units, with `n` always reported.
- **Secondary:**
  - pooled RMSE, normalised MAE (z-units), sensor-macro MAE and RMSE;
  - per client: micro and macro MAE and RMSE for the 4 geographic regions;
  - per horizon: 5 to 60 min;
  - per sensor: MAE distribution (median, IQR, 5–95%), per type group and flow tertile, and sensor win counts between arms. Distributions only, no IDs.
- **Exploratory, by day:** per test day; weekday (Thu, Fri, Mon, Tue) versus weekend (Sat, Sun).
  - The weekend is the main new condition: training covered three weekends and two holidays, but validation had no weekend.
- **Negative-prediction rate:** unclipped, per arm, overall and on weekend days.
- **Reference forecasts:** persistence, daily seasonal naive and historical average (training-only time-of-day medians), on the same test windows and masks.
- **Validation-to-test change per arm:** pooled MAE difference and ratio. Descriptive only; this is the generalisation gap of each frozen model.

## 6. Statistical analysis (the same as S5; only the data change)
- **Contrasts:**
  - C1 = geographic FedAvg − centralised;
  - C2 = geographic FedAvg − local-only;
  - C3 = IID FedAvg − geographic FedAvg;
  - C4 = local-only − centralised.
- **Pairing:** on identical test cells, with seeds averaged per cell.
- **Primary uncertainty:**
  - **resampling:** whole sensors within each geographic client, 2,000 times, **bootstrap seed 20261028** (unchanged from S5);
  - **spatial sensitivity:** a freeway-direction cluster bootstrap (83 clusters, the same as S5);
  - **reported values:** the wider interval and the larger p (two-sided, floored at 1/2000);
  - **multiplicity:** Holm across C1–C4.
- **Practical threshold:** SESOI = **0.1638** (fixed from validation).
- **Wording rules (as S5, with "test"):**

| Situation | Wording |
|---|---|
| Interval excludes 0, Holm p < 0.05 **and** \|Δ\| ≥ 0.1638 | "supported on January test data" |
| Interval excludes 0, but \|Δ\| < 0.1638 or Holm p ≥ 0.05 | "detectable but below the practical threshold on January test data" |
| Otherwise | "not distinguishable on January test data" |

- **Sensitivity:**
  - leave-one-test-day-out (6 point estimates per contrast);
  - weekday versus weekend contrasts (exploratory);
  - matched seeds 1–2 only (dropping the capped geographic seed-0 run, as in S5).
- **Do test results agree with validation?** For each contrast, report whether the test verdict matches the validation verdict, side by side.
  - **Wording:** a contrast is "consistent with validation" only when the two verdicts match.
  - **No pooling:** validation and test p-values are never combined.
- **Exploratory:** per client, per horizon, per type group, per tertile and per day. Intervals only, with no claims and no multiplicity control.

## 7. Expected runtime and resources (estimated)

| Step | Estimate | Basis |
|---|---|---|
| E2 validation re-prediction (21 models) | about 20–30 s | 0.8 s per 170,600 windows at S4 |
| E3 test inference (21 models, 343,246 windows) | about 30–60 s | about 2× the validation size |
| E3 reference forecasts and metrics | about 5–10 s | S4 `--baselines` timing |
| E4 bootstraps and report | about 5–10 s | S5 ran in 3 s on half the cells |
| **Total** | **under about 3 min** of compute, plus checks | |
| Peak memory | about 1–1.5 GB | twice the validation arrays, with every run's predictions held for the pairing |
| Disk | about 0.35 GB (12 test prediction records of about 28 MB, plus metrics) | twice the S4 validation records |

## 8. Risks and how they are handled

| Risk | Handling |
|---|---|
| Accidental early look at test data | One-shot marker. Hash-gated script. E2 touches validation only. No other code path loads test windows (`run_phase7.py` has no test option; `run_phase6.py --evaluate test` would **retrain** and is not used) |
| Frozen checkpoint is not the S4 best model | E2 requires byte-identical reproduction of the saved validation predictions before E3 |
| Weekend distribution shift | Reported per day and weekday/weekend. A larger test error than validation is expected and is not a defect |
| Validation optimism, then a small test effect | Test verdicts may differ from validation. Both are reported and nothing is re-tuned |
| C1 interval near the SESOI | The pre-declared wording decides. No post-hoc threshold changes |
| Negative predictions on low-flow weekend periods | Reported unclipped, as fixed in Phase 6 |
| Missing test targets (290 missing cells, 1,566 masked target cells at S2) | Masked from every metric, with `n` reported |
| Provisional geography | Every per-client, per-type and per-District statement is labelled provisional |
| Execution defect after E3 starts | Stop, preserve outputs, report. A rerun only with approval and full disclosure |
| A test result that disappoints | Final for January Phase 8. Any new modelling becomes a new, separately approved phase evaluated on new data (for example later months), never on this test split again |

## 9. Decisions for approval
| # | Decision | Recommendation |
|---|---|---|
| T-1 | This protocol as the pre-registration. Its SHA-256 at approval becomes `--protocol-sha256` | approve |
| T-2 | Frozen set = the manifest `721b2c94…d505` (12 runs, 21 checkpoints, configs, geo200 checkpoint, code, SESOI) | approve |
| T-3 | SESOI 0.1638 and the S5 statistics unchanged, with bootstrap seed 20261028 | approve |
| T-4 | Local-only routing: each client's model predicts only its own sensors | approve |
| T-5 | E1 (script plus synthetic tests) as the next stage; E2–E4 each gated | approve |
| T-6 | Commit policy: protocol, script, tests and aggregate report may be committed with approval. Predictions and per-sensor outputs never | confirm |
