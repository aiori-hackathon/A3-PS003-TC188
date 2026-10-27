# Phase 7 ablation plan — FedAvg client optimiser state

**Date:** 2026-10-27. **Status:**
- Implementation and smoke tests are done.
- **The six full runs are PROPOSED and not started.**
- The ≈ 200-sensor outline (§7) is FUTURE; no data has been prepared.

**Question:** does re-initialising each client's Adam optimiser at every round (`reset_each_round`, Phase 7 P7-5) contribute to the FedAvg performance gap seen in January?

**January pilot validation results** (seeds 0–2; MAE in vehicles per 5 minutes):

| Arm | MAE |
|---|---|
| Centralised B1 | 19.743 ± 0.093 |
| Local-only | 19.872 ± 0.085 |
| Geographic FedAvg | 20.158 ± 0.047 |
| IID-like FedAvg | 20.228 ± 0.017 |

The IID-versus-geographic difference is **not** statistically established.

---

## 1. What changed

| Item | Change |
|---|---|
| `aiori/federated.py` | `Client` reads `cfg["optimizer_state"]`. It may be `reset_each_round`, which is the default and also applies when the key is absent, or `persist_per_client`; any other value raises. `local_train` is described below. Also new: `adam_step_count()` for logs and tests; `stats[-1]["adam_step_total"]` in persist mode. `Server`, aggregation, partitions, the training loop and validation are unchanged |
| `experiments/run_phase7.py` | the output root comes from `--out-dir`, else from the config's `results_subdir` (default `phase7`, which is unchanged for the existing configs). It **refuses to write into a non-empty run folder**, checked before any data is loaded |
| New configs | `experiments/configs/phase7_ablation_fedavg_geo_b1_persist.json` and `phase7_ablation_fedavg_iid_b1_persist.json`. Copies of the Phase 7 FedAvg configs that differ only in `arm`, `description`, `optimizer_state: "persist_per_client"`, `results_subdir: "phase7_ablation"` and `ablation_of` |
| Unchanged | original Phase 7 configs (same SHA-256), Phase 6 modules, preprocessing, selections, and all Phase 6 and Phase 7 results |

**How `local_train` behaves in each mode:**
- **Reset:** it keeps the original behaviour, a fresh model and a fresh Adam on every call.
- **Persist:** each client keeps **one private model object and one private Adam** for the whole run. At the start of every round the global weights are copied in place with `load_state_dict`, so the parameters equal the global model while Adam's moments and step count carry over.
- **Optimiser state never leaves the client:** it is never aggregated, shared or transferred. The server's interface rejects anything other than model weights.

## 2. Verification completed

**Tests: 144 passed**, of which 11 are new in `tests/test_federated_optimizer.py`:

| Test | Shows |
|---|---|
| default is reset; unknown value rejected | backward compatibility |
| explicit `reset_each_round` = config without the key | identical weights, byte logs and validation curves |
| reset mode holds no optimiser between rounds | no hidden state |
| persist = one manual Adam across two rounds, with weights re-synchronised between rounds | exact weights; Adam step count 6 after 2 × 3 batches |
| first round identical in both modes; second round differs | the modes differ only through carried state |
| every round starts from the global parameters | synchronisation (parameters at the entry of local training equal the global state) |
| optimiser state private per client | separate optimiser and model objects, no shared tensor memory; the server rejects an optimiser `state_dict`; the server holds no optimiser |
| persist FedAvg aggregation | the server state equals the weighted average of the received client states in every round; byte log identical to reset mode; Adam steps accumulate per client (one round's batches, then twice that) |
| persist determinism | same seed gives identical results; different seed gives different results |
| single-client persist FedAvg = continuous centralised training | exact equality over 2 rounds |
| runner refuses to overwrite results | checked before data loading |

**Smoke runs** (seed 0; 2 rounds × 5 batches per client; validation only; outputs in `results/phase7_ablation/`):

| Setup | Arm | Val MAE, round 1 → 2 | Adam steps per client after round 1 / 2 |
|---|---|---|---|
| Primary | geographic, persist | 85.068 → 80.152 | 5 / 10 |
| Primary | IID, persist | 85.092 → 80.172 | 5 / 10 |
| Missing-data | geographic, persist | 13.145 → 12.399 | 5 / 10 |
| Missing-data | IID, persist | 13.143 → 12.394 | 5 / 10 |

- **Health checks:**
  - weights and predictions finite; no skipped batches; no clipping;
  - client error sums equal the merged prediction records;
  - 475,304 training bytes per round, as in reset mode;
  - partitions identical to the original Phase 7 partitions (SHA-256).
- **Round 1 equals the reset runs exactly**, as expected. These smoke numbers are a pipeline check only.

**Reset-mode reproduction:** the original reset configs were rerun in smoke mode into `results/phase7_ablation/reset_reproduction/`. Predictions, checkpoints, histories, all metrics and communication counts are **identical** to the original `results/phase7/*_smoke` runs, on both the primary and missing-data runs.

**Integrity:** 175 fingerprints were unchanged before and after the smoke runs:
- data files;
- size and modification time of the 12 monthly raw files (the 11 non-January files were not opened);
- all Phase 6 and Phase 7 results;
- code and configs.

## 3. Proposed full runs (not started)

| Run | Config | Seeds | Output |
|---|---|---|---|
| Geographic FedAvg, persistent client Adam | `phase7_ablation_fedavg_geo_b1_persist.json` | 0, 1, 2 | `results/phase7_ablation/primary/fedavg_geo_b1_persist_seed{0,1,2}/` |
| IID-like FedAvg, persistent client Adam | `phase7_ablation_fedavg_iid_b1_persist.json` | 0, 1, 2 | `results/phase7_ablation/primary/fedavg_iid_b1_persist_seed{0,1,2}/` |

- **Identical to Phase 7:** B1 architecture, partitions (same SHA-256), batch 512, lr 1e-3, clipping 1.0, E = 1, R_max = 50, patience 5, `n_k` weights, float64 aggregation, seeds, client shuffle streams, and the validation protocol.
- **References (reused, not rerun):** the Phase 7 reset runs `results/phase7/primary/fedavg_{geo,iid}_b1_seed{0,1,2}`. Local-only and centralised B1 are reused as context.
- **Runtime estimate:** about 4 s per round. The reset runs took 81–171 s each, so the six runs need about 10–20 min of CPU, at most 6 × 3.5 min ≈ 21 min.
- **Memory:** plus about 0.2 MB of Adam state across 4 clients.
- **Execution:**
  - sequential;
  - fingerprints before and after (data, raw files, Phase 6 and Phase 7 results, code, configs);
  - the runner's overwrite guard stays on;
  - completion confirmed from files on disk.

## 4. Comparison plan (pre-declared)

**Primary contrast:** persist − reset within each partition, paired by seed (same initial weights, partition and shuffle streams).

| Report | Content |
|---|---|
| Pooled metrics | MAE and RMSE per seed, mean ± SD; paired persist − reset differences per seed, mean ± SD; normalised MAE; negative-prediction rate |
| Gap closed | for geographic FedAvg: (MAE_reset − MAE_persist) / (MAE_reset − MAE_local) and / (MAE_reset − MAE_central), per seed and as a mean. Reported descriptively |
| Per horizon | MAE and RMSE at 5 … 60 min; paired differences at 5, 30, 45 and 60 min |
| Per client | micro MAE and RMSE for the four geographic regions; paired differences; per-sensor counts of persist < reset |
| Convergence | validation MAE per round (persist vs reset overlaid per seed); first round within 1% of the arm's best; first round below historical average (25.779) |
| Selection | best round, rounds run, early-stopping flag |
| Training budget | optimiser steps per round and in total; Adam step count per client at the best round; clipped steps; largest gradient norm |
| Communication | training bytes per round, in total and to the best round; evaluation downloads reported **separately**. Persist adds **no** bytes because optimiser state is never transmitted |
| Secondary contrast | IID − geographic within the persist mode, reported like Phase 7 |

**Wording rules:**
- "Consistent across seeds" is used **descriptively**, only when all three paired differences have the same sign.
- No significance claims.
- The IID-versus-geographic difference is not described as established.

**Pre-declared interpretation:**
- **If persist lowers geographic FedAvg MAE on all three seeds:** report that the reset contributes to the gap, together with the fraction closed.
- **Otherwise:** report that this pilot shows no consistent contribution.

**Selection discipline:**
- The ablation reuses the same 3-day validation set that selected B1 and stopped every run. Its results are diagnostic.
- Making `persist_per_client` the default for later phases would be a validation-based choice and needs separate approval.
- The test split stays closed.

**Limitations to repeat:**
- The 40-sensor pilot is underpowered (≈ 17% power at 10 sensors per client) and its clients are weakly heterogeneous.
- The station-ID mapping is provisional (SUPPORTED BY STRONG INDIRECT EVIDENCE; UNCONFIRMED BY PROVENANCE).
- Validation covers three weekdays.
- The pilot is a completeness-selected subset.
- This is a simulation, not a deployment.

## 5. Decisions requesting approval

| # | Decision | Recommendation |
|---|---|---|
| AB-1 | Run the six ablation runs above (validation only) | approve when ready |
| AB-2 | Reuse the Phase 7 reset runs as the paired reference (no rerun) | approve; the reset path was shown to reproduce them exactly in smoke mode |
| AB-3 | Comparison and wording rules in §4 | approve |

## 6. Files

**New:**
- `tests/test_federated_optimizer.py`
- `experiments/configs/phase7_ablation_fedavg_geo_b1_persist.json`
- `experiments/configs/phase7_ablation_fedavg_iid_b1_persist.json`
- this document

**Changed:** `aiori/federated.py` and `experiments/run_phase7.py` (§1).

**Outputs:** under the git-ignored `results/phase7_ablation/`.

Nothing has been committed.

---

## 7. Outline: ≈ 200-sensor geographic experiment (FUTURE; nothing prepared)

**Purpose:** give the IID-versus-geographic and federated-versus-local comparisons enough power. R2 power is about 0.92 at 50 sensors per client.

| Step | Outline (each needs approval) |
|---|---|
| Data | January 2023 only (the same raw file, `2023_p01.npy`), training-period eligibility (A1), exact-duplicate exclusion (A2). Eligible population: 11,188 sensors. No other months |
| Selection policy | ≈ 50 sensors per geographic client (≈ 200 in total). Seeded and **stratified within each client** by sensor type (Mainline / ramp / other) and by training-period mean-flow tertile. This reduces the completeness bias of the 40-sensor pilot (for example, coastal pilot median flow 276 vs 73 in the population). Keep the District floor rule where feasible. Training-period statistics only; the policy is written down before selection |
| Preprocessing | a new checkpoint in a separate folder (e.g. `data/processed/geo200/`). Same split, per-sensor training-only scaler, causal fill, observed flag, masked targets. `prepare_pilot` would need `--max-sensors 250` (default limit 50); no code change to the frozen pipeline. Integrity checks and `verify_artifacts`-style checks for the new run |
| Clients | the same 4 geographic clients (`proposed_client`), plus an IID-like sensor partition of the same 200 sensors (fixed partition seed) |
| Arms | centralised, local-only, IID and geographic FedAvg (reset, or the ablation winner if approved) × seeds 0–2; validation only |
| Compute (estimate) | about 5× the pilot windows: about 19 s per epoch or round on CPU, 6–15 min per run, about 2–3 h for 4 arms × 3 seeds. Arrays about 35 MB. Communication per round is unchanged (model-size bound) |
| Limitations | still January only and provisional station IDs; validation still 3 weekdays. Stratification makes clients more representative but does not remove the ID caveat |
| Not included | full-year 2023, test evaluation, new partitions beyond geographic and IID-like |

---

## 8. Results of the six ablation runs (2026-10-27; validation only)

**Runs:** AB-1 to AB-3 were approved on 2026-10-27. All six runs completed (exit 0); completion was confirmed from the files on disk. The 215 pre-existing fingerprints were unchanged, and the only new files are in the six run folders.

**Checks:**
- All 18 compared runs used the same validation targets.
- Each persist config differs from its reset reference only in `optimizer_state` (plus labels). Partitions and seeds are identical.
- All predictions and checkpoints are finite, and client error sums match the merged records.

| Arm (seeds 0–2) | MAE | RMSE | Best round | Rounds run | Training MB (to best) | Eval-download MB |
|---|---|---|---|---|---|---|
| Centralised B1 | 19.743 ± 0.093 | 31.807 ± 0.142 | (epochs 18, 18, 24) | | 0 | 0 |
| Local-only | 19.872 ± 0.085 | 31.959 ± 0.135 | | | 0 | 0 |
| Geographic, reset | 20.158 ± 0.047 | 32.470 ± 0.108 | 39, 36, 20 | 44, 41, 25 | 17.43 (15.05) | 8.71 |
| Geographic, persist | 20.251 ± 0.083 | 32.640 ± 0.094 | 27, 15, 24 | 32, 20, 29 | 12.83 (10.46) | 6.42 |
| IID, reset | 20.228 ± 0.017 | 32.592 ± 0.046 | 28, 15, 22 | 33, 20, 27 | 12.68 (10.30) | 6.34 |
| IID, persist | 20.185 ± 0.054 | 32.526 ± 0.087 | 27, 23, 22 | 32, 28, 27 | 13.78 (11.41) | 6.89 |

**Paired persist − reset differences:**

| Partition | MAE per seed | Mean ± SD | RMSE mean | Direction |
|---|---|---|---|---|
| Geographic | +0.001, +0.231, +0.049 | +0.093 ± 0.121 | +0.170 | all three ≥ 0, so persist is not better; seed 0 is a tie |
| IID | −0.061, −0.103, +0.034 | −0.043 ± 0.070 | −0.066 | mixed direction |

- **Gap closed** (geographic, persist vs reset): −30% of the gap to local-only, i.e. the gap widened slightly.
- **Persist versus local-only and centralised:** still worse on all three seeds in both partitions.
- **Convergence:** persist is slower in early rounds. Mean validation MAE at round 2 was 22.51 vs 22.04, and at round 5 21.14 vs 20.94. The two catch up later.
- **Cost:** communication per round is unchanged (no optimiser bytes) and so is time per round (about 3.84 s).

**Conclusion (descriptive; no significance claim):**
- Persisting client Adam state **does not consistently improve** FedAvg on this pilot.
- The reset (P7-5) therefore does not explain the gap between FedAvg and local-only or centralised training.
- `reset_each_round` remains the default.
- The 40-sensor pilot is underpowered, and this is a validation-only diagnostic.
