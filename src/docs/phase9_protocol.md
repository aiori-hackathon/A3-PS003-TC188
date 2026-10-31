# Phase 9 protocol — nine-month (Jan 1 – Sep 30, 2023) autonomous experiment

**Written:** 2026-10-28, before any nine-month model was trained and before the nine-month test period was evaluated.

**Authority:**
- The project lead's final execution order of 2026-10-28 (autonomous, 16-hour deadline).
- The final one-time test is authorised in that order.

**Runner:** `experiments/phase9/`. It is a standalone controller, worker, final stages and watchdog, and needs no Claude Code, network or user input. The freeze stage records this document's SHA-256 before the test.

**Hard deadline:** 2026-10-29 03:38:10 UTC (16 hours after start). The training cutoff is 90 minutes earlier, at 02:08:10 UTC; the remaining time is reserved for freeze, pre-check, test, analysis and reports.

---

## 1. Scope and separation from January (Phase 8)
- **Phase 8 is untouched:** its 12 runs, 21 checkpoints, protocol and test results are not modified, rerun or retrained. The integrity report re-hashes them at the end.
- **Data:** XTraffic flow (channel 0), files `2023_p01` to `2023_p09` only. October–December are never opened; the code refuses them.
- **Split** (chronological, verified programmatically from the time origin 2023-01-01 00:00 at 5-minute steps):
  - train 1 Jan – 10 Jul (191 days, steps [0, 55008));
  - validation 11 Jul – 7 Aug (28 days, [55008, 63072));
  - test 8 Aug – 30 Sep (54 days, [63072, 78624)).
- **Task:** 12 past steps → 12 future steps.
  - Per-sensor z-score fitted on the training period only.
  - Causal input fill: forward fill of at most 6 steps within the split, then the training time-of-day median, then the training mean.
  - An observed flag is kept as a feature; missing targets are masked.
  - **A window is kept only if both its inputs and its targets lie inside one split.** This is stricter than Phase 8, which let inputs reach into the previous split.
  - Metrics are MAE and RMSE in raw units (vehicles per 5 min).

## 2. Data integrity (stages audit, prepare, verify)
- **Shape:** every month is (16972, days × 288, 3) float32, with a complete calendar: 273 days, no missing or duplicated hour.
- **Sensor order across months:** rank correlation of per-sensor monthly mean flow between consecutive months (> 0.9 required), plus continuity at month boundaries.
- **Daylight saving (12 Mar):** the alignment of the daily profile in the week before and after is reported.
- **Eligibility (training period only):** not fully missing, not constant, not an exact copy (all members of a SHA-256 copy group excluded), coverage ≥ 99%, zero readings ≤ 50%. These are the approved Phase 8 rules applied to the 191 training days.
- **Selection:**
  - The January geo200 sensors that are still eligible are kept.
  - Each one that is not is replaced deterministically, using salted-hash order (`aiori-ninemonth-v1`) and training-period statistics only. The replacement comes from the same client, preferring the same type group and nine-month flow tertile, then the same type, then any type.
  - Pilot sensors and January geo200 sensors are never used as replacements.
  - Result: 50 sensors per client.
- **Verification:** shapes, source equality, train-only scaler and medians, mask equals missing values, finite inputs, every window inside its split, each window with at least one observed target, and loader checks. A failed check stops the experiment; the result is then an honest partial-delivery report.

## 3. Training (stage train)
- **Arms:** centralised GRU, geographic FedAvg, local-only GRU, IID FedAvg.
- **Settings:** the Phase 8 B1 settings unchanged:
  - GRU(4 → 64) → 12;
  - Adam, lr 1e-3, batch 512, clip 1.0;
  - at most 50 epochs or rounds, patience 5 on pooled validation MAE;
  - FedAvg with E = 1, full participation, `reset_each_round`, IID partition seed 20261027;
  - 8 threads per process.
- **Configs:** `experiments/configs/phase9_*_b1.json`.
- **Resumable training:** `aiori/resumable.py` is operation-for-operation identical to the Phase 8 loops (tested). It checkpoints model, optimiser, shuffle generator, RNG, epoch or round, best state, early-stopping state and history after every epoch or round, so an interrupted run resumes exactly.
- **Priority:**
  1. centralised seed 0;
  2. geographic FedAvg seed 0;
  3. local-only seed 0;
  4. IID FedAvg seed 0;
  5. then seeds 1 and 2 in the same arm order.
- **Parallelism:** decided by a measured benchmark (1, 2 or 3 concurrent processes; more than one only if total throughput is at least 15% higher). Thread counts are fixed, so results don't depend on parallelism.
- **Deadline policy:**
  - A run is launched only if its estimated duration fits before the training cutoff. Centralised and geographic FedAvg seed 0 are always attempted.
  - Runs still going at the cutoff are stopped and recorded as incomplete; **they are never reported or tested.**
  - Budgets are never reduced.

## 4. Validation-only choices (fixed by the freeze stage before the test)
- **Checkpoint:** every completed run's best-validation checkpoint (restored automatically). There is no other selection within a run.
- **Selected final model:** the arm with the lowest seed-averaged validation MAE (RMSE breaks ties).
- **Clipping negative predictions at 0:** adopted for an arm only if it lowers that arm's validation MAE. The primary, comparable results are always reported **unclipped**; clipped results are a secondary, validation-selected variant.
- **SESOI:** 1% of the mean validation MAE of the completed centralised runs. If no centralised run completed, 1% of the mean validation MAE of all completed runs.
- **Not done:** no learning-rate, regularisation or feature changes were tried. Time is reserved for the primary comparable experiment; this is recorded as a limitation.

## 5. Frozen evaluation (stages freeze → precheck → test → analysis → report)
1. **Freeze:** SHA-256 of every completed run file, the dataset, the code (aiori and the runner), the configs and this document. It refuses if any test output exists.
2. **Pre-check (validation only):** reload every frozen checkpoint, re-predict validation with the training routing, and require byte-identical agreement with the saved validation predictions.
   - **Routing:** centralised uses one model; local-only uses each client's own model on its own sensors; FedAvg uses the global model, applied per partition client.
3. **Test, once:**
   - `RUN_STARTED` is written before any test window is loaded. The run covers all frozen runs and the three reference forecasts (persistence, historical average, same time yesterday).
   - **Recovery procedure:** if a test pass is interrupted, its folder is preserved unchanged as `test_partial_attempt_N`, and the pass is re-run from the same frozen models (inference is deterministic), at most twice.
   - Partial outputs are never analysed or used for any decision.
4. **Analysis (pre-declared, Phase 8 method):**
   - **Contrasts:** C1 = geographic FedAvg − centralised; C2 = geographic FedAvg − local-only; C3 = IID − geographic FedAvg; C4 = local-only − centralised. Each uses the seeds both arms completed (matched) and is paired per cell.
   - **Uncertainty:** a sensor-cluster bootstrap within geographic clients (2,000 replicates, seed 20261028) and a freeway-direction cluster sensitivity. The wider interval and the larger p are reported, with Holm adjustment over the available contrasts.
   - **Wording:** as in Phase 8, distinguishing statistical detectability (interval excludes 0 and Holm p < 0.05) from practical meaning (|Δ| ≥ SESOI).
   - **Single matched seed:** reported as sensor-sampling uncertainty only, never as a three-seed result.
5. **Report:**
   - test MAE, RMSE and normalised MAE;
   - improvement over the reference forecasts;
   - per client, per horizon, August vs September, weekday vs weekend;
   - negative-prediction rate and clipped variants;
   - validation → test change;
   - runtime and communication;
   - figures, reproducibility manifest, integrity report, and the task log of completed, failed and skipped tasks.

## 6. Autonomy, safety and recovery
- **Controller:** single instance (OS file lock), atomic state file, 20-second heartbeat, bounded retries with exponential backoff (30, 120, 300 s), idempotent stages, completed-stage skipping. Integrity failures are never retried blindly.
- **Watchdog:** a per-user scheduled task (`AIORI_PS003_Phase9`), triggered at logon and every 5 minutes, runs `watchdog.py`. It restarts a dead controller, kills a hung one (no heartbeat for 20 minutes), allows at most 6 restarts per hour and 40 in total, and unregisters itself on completion.
- **Sleep and permissions:** sleep is prevented with `SetThreadExecutionState` while the controller runs. No system or security setting is changed, and no elevation is used.
- **Privacy:** all sensor-level data, manifests, checkpoints and predictions stay under git-ignored `data/` and `results/`. Documents are aggregate only.
