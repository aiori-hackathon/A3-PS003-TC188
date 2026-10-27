# Phase 8 technical specification — ≈ 200-sensor geographic federated experiment

**Date:** 2026-10-27. **Status: PROPOSED (design only).**

**Not done yet:**
- no sensors selected;
- no preprocessing artifacts created;
- no code written;
- no training.

**Labels:**
- **[approved]:** approved in writing by the project lead (source given).
- **[proposed]:** awaiting approval.
- **[measured]:** observed in completed project runs.
- **[estimate]:** extrapolated.

**Companion documents:**
- `docs/phase8_200_sensor_implementation_plan.md`: stages, checklist, files, decisions.
- `docs/research_validity_decisions.md`: R1/R2, decisions A1–A8.
- `docs/phase7_technical_spec.md`, `docs/phase7_implementation_plan.md`, `docs/phase7_ablation_plan.md`.

**Sources reviewed for this design (read-only):**
- code: `aiori/flow_audit.py` (eligibility, duplicate rule, `select_pilot`), `aiori/prepare_pilot.py`, `aiori/preprocessing.py`, `aiori/windows.py`, `aiori/dataset.py`, `aiori/federated.py`, `experiments/run_phase6.py`, `experiments/run_phase7.py`;
- configs: `experiments/configs/phase6_b1.json`, `phase7_*.json`;
- Phase 6 and Phase 7 result files under `results/`;
- the frozen audit tables `data/processed/flow_sensor_audit.csv` and `sensor_metadata_aligned.csv`. These were used for **aggregate** feasibility counts only (§2.6); no sensor was selected and no ID was exported.

---

## 0. Background and what carries over

| Item | Value | Status |
|---|---|---|
| Data | XTraffic `year_2023/2023_p01.npy`, Channel 0 (flow), January 2023 only | [approved] |
| Split | train [0, 6336) = 1–22 Jan; val [6336, 7200) = 23–25 Jan (Mon–Wed); test [7200, 8928) = 26–31 Jan | [approved] A5 |
| Eligibility (training period only) | not fully missing; not constant; not an exact-copy series (all members of training-period SHA-256 copy groups excluded, A2); coverage ≥ 99%; zero readings ≤ 50% → **11,188 eligible** of 16,972 | [approved] A1, A2 |
| Clients | `CLIENT_BY_DISTRICT`: LA/Ventura = D7; Bay Area = D4; Coastal/South Central = D5 + D12; Inland/Central = D3 + D6 + D8 + D10 | [approved] A7 |
| Model | B1 `ForecastGRU(4, 64)`: z-flow, observed flag, time-of-day sin/cos; 14,220 parameters | [approved] B1 adoption |
| Training | Adam lr 1e-3, batch 512, clip 1.0, no LR schedule, ≤ 50 epochs/rounds, patience 5 on pooled validation MAE (raw units) | [approved] D-C, D-D |
| FedAvg | E = 1, full participation, `n_k` = training windows, float64 aggregation, **`reset_each_round`** | [approved] Phase 7; reset retained after the ablation (2026-10-27) |
| Row-to-ID mapping | **SUPPORTED BY STRONG INDIRECT EVIDENCE; UNCONFIRMED BY PROVENANCE** | [approved] A6 |

Every District, County, Type and client label is attached through `node_order[array_index]` and is therefore **provisional**. Nothing in Phase 8 verifies the mapping.

**Pilot results that motivate Phase 8** (40 sensors, validation, seeds 0–2, MAE):

| Arm | MAE |
|---|---|
| Centralised | 19.743 ± 0.093 |
| Local-only | 19.872 ± 0.085 |
| Geographic FedAvg | 20.158 ± 0.047 |
| IID FedAvg | 20.228 ± 0.017 |

- The pilot is completeness-selected: every sensor has 100% training coverage, all 40 are Mainline or HOV, and there are no ramps.
- It is underpowered (≈ 17% power to detect the R2 client heterogeneity at 10 sensors per client).

---

## 1. Research questions and hypotheses

| # | Question | Pre-declared hypothesis | Basis |
|---|---|---|---|
| RQ1 | What does federation cost relative to centralised training under the geographic partition? | H1 (directional): geographic FedAvg pooled MAE > centralised | pilot +0.41 on all 3 seeds |
| RQ2 | Does collaboration help relative to isolated clients? | H2 (**two-sided**, no direction declared) | pilot: FedAvg worse than local-only (+0.29), but each local model then had only 10 sensors; at 50 per client local data grows 5× |
| RQ3 | Does the geographic partition differ from an IID-like partition of the same sensors? | H3 (**two-sided**) | pilot difference +0.07 with mixed sign; R2: clients weakly heterogeneous (η² ≤ 0.10) |
| RQ4 | Do the pilot's arm orderings replicate on a stratified, less completeness-biased sample? | descriptive | — |
| RQ5 | How heterogeneous are the four selected client samples? | descriptive (training-period features, R2 method) | R2 power 0.92 at 50 per client |

**Power caveat:** the R2 power figure (≈ 0.92 at 50 per client) is the power to detect **descriptive client heterogeneity** in training-period features. It is **not** a power calculation for differences in forecast error between arms, and no such calculation exists. Phase 8 makes no promise of statistical significance.

---

## 2. Sensor selection policy

### 2.1 Inputs (read-only, frozen)
- `data/processed/flow_sensor_audit.csv`: training-period statistics, `eligible`, `exclusion_reason` (produced by `aiori/flow_audit.py` at checkpoint `f45bc9a`).
- `data/processed/sensor_metadata_aligned.csv`: provisional `Type`, `Fwy`, `Direction`, District.
- `data/processed/sensor_selection.csv` and `missing_validation_selection.csv`: used **only** to exclude the 48 pilot sensors.
- The SHA-256 of each input is recorded. The selection refuses to run if any differs from the hash recorded at approval time.

**Columns allowed in selection:** `array_index`, `proposed_client`, `District`, `eligible`, `mean` (training-period mean flow), and metadata `Type`.

**Columns not used:** every `month_*` column and `eval_missing_count`, which contain validation/test information. Validation/test values, missingness and model results are never read.

### 2.2 Candidate set
1. `eligible == True` (A1, A2, unchanged): 11,188 sensors.
2. Exclude the 40 primary-pilot and 8 missing-data-validation sensors [proposed, D8-1]: **11,140 candidates**.
   - **Why:** the new sample is independent of the pilot, whose validation results have already been inspected. The pilot stays a separate, unchanged testbed.
3. Each sensor has exactly one client because the client is a function of the District. A uniqueness check is applied anyway.

### 2.3 Strata [proposed, D8-2]
- **Sensor-type group** (from provisional `Type`):

  | Group | Types |
  |---|---|
  | `mainline` | Mainline |
  | `hov` | HOV |
  | `ramp` | On Ramp, Off Ramp |
  | `other` | Fwy-Fwy, Coll/Dist, Conventional Highway |

- **Flow level:** tertiles of training-period mean flow, computed **within each (client, type group)**. The ranking uses `rank(method="first")` before `qcut`, so ties are handled deterministically.
  - **Why within type:** type and flow are strongly confounded. Almost all ramps fall in the lowest client-wide tertile. Within-type tertiles keep every type's low, medium and high range represented.

### 2.4 Allocation and draw [proposed, D8-2, D8-3]
1. **Per client:** 50 sensors.
2. **Across type groups:** proportional to the client's candidate counts, using the largest-remainder method. Ties in remainder go to the larger group first, then alphabetically.
3. **Across flow tertiles within a type group:** k // 3 to each tertile. The remainder goes to tertiles in the fixed order mid, high, low.
4. **Within each (client, type group, tertile) stratum:**
   - candidates are ordered by `sha256("aiori-geo200-v1:<array_index>")`, which is the seeded, version-independent hash ordering already used by `select_pilot`;
   - the first k are taken;
   - the salt `aiori-geo200-v1` is the documented seed.
5. **Missing count is not used for ranking.** The pilot's `missing_count`-ascending rank is what caused the completeness bias.
6. **Fallbacks:**

   | Problem | Rule |
   |---|---|
   | A stratum is short | take all; move the shortfall to the other tertiles of the same type group (mid, high, low order), then to the other type groups by the next largest remainder |
   | A District in the client gets 0 sensors | apply the A3 floor: replace the last-drawn sensor of the client's largest stratum with the first-hashed candidate of the missing District from the same type group (if none, from any group) |

   Every fallback is logged with counts.

### 2.5 Missing observations
- Eligibility already requires ≥ 99% training coverage. Within that, sensors with gaps are **not** penalised.
- Inputs use the frozen causal fill: forward fill ≤ 6 steps within a split, then the training time-of-day median, then the training mean. An observed flag is kept as a feature.
- Targets are never filled. Missing targets stay NaN and are masked from loss and metrics. Windows whose 12 targets are all missing are dropped (frozen rule).
- Validation/test missingness of the selected sensors is **unknown by design** until preprocessing, and is reported then. It is not a selection criterion.

### 2.6 Feasibility [measured 2026-10-27; aggregate counts from the frozen audit; no draw performed]

| Client | Candidates | Expected allocation (mainline / ramp / hov / other) | Districts (candidate share) |
|---|---:|---|---|
| LA/Ventura | 2,322 | 27 / 13 / 9 / 1 | D7 (100%) |
| Bay Area | 3,096 | 36 / 13 / 0 / 1 (only 2 HOV candidates) | D4 (100%) |
| Coastal/South Central | 2,132 | 30 / 7 / 11 / 2 | D12 (80%), D5 (20%) |
| Inland/Central | 3,590 | 35 / 8 / 6 / 1 | D8 (31%), D3 (28%), D10 (23%), D6 (18%) |

- **Shortfalls:** every allocated stratum has ≥ 11 candidates per tertile, so none is expected. With these District shares, the District floor is not expected to bind.
- **Coverage:** 22.6% of candidates have < 100% training coverage. Their training gaps are at most 8 steps at the 90th percentile and 11 at the 99th.
- **Contrast with the pilot:** the pilot had 100% coverage, 35 Mainline + 5 HOV, and no ramps.

### 2.7 How this helps the geographic non-IID question (no significance promised)
- **More sensors per client** (10 → 50): power for descriptive client heterogeneity rises from ≈ 0.17 to ≈ 0.92 (R2). Each client's error estimates rest on 5× as many sensors.
- **Representative composition:** proportional stratification by type and flow level makes each client sample resemble its eligible population. That includes ramps and low-flow sensors. The pilot over-represented complete, high-flow mainline sensors (for example, Coastal pilot median flow 276 vs 73 in the population).
- **Same sensors in every arm:** the IID-like partition reshuffles the **same** 200 sensors, so IID versus geographic isolates the grouping effect.
- **Still not verified:** heterogeneity remains defined by provisional geography.

### 2.8 Outputs (local only; git-ignored under `data/processed/`)

| File | Content |
|---|---|
| `data/processed/geo200/sensor_selection.csv` | selection manifest. Same columns as the pilot manifest (so `prepare_pilot.load_selection` accepts it), plus `type_group`, `flow_tertile`, `stratum`, `selection_rank`, `selection_basis` (`stratified_hash` / `fallback_*` / `district_floor`) and `selection_salt` |
| `data/processed/geo200/selection_summary.json` | input SHA-256s, salt, rules, per-stratum candidate, allocated and drawn counts, fallbacks, District counts, totals |
| `data/processed/geo200/selection_report.md` | human-readable summary. Contains sensor-level IDs, so it **stays local** |

---

## 3. Preprocessing checkpoint

**Same frozen pipeline, separate output:** `aiori/prepare_pilot.py`, unchanged, is run with:

```
--selection data/processed/geo200/sensor_selection.csv --out-dir data/processed/geo200
--min-sensors 150 --max-sensors 250 --run-label "geo200 (Phase 8)"
```

All other defaults are unchanged: per-sensor scaler, max-ffill 6, input/horizon 12/12, fractions 0.7/0.1/0.2, 288 steps per day.

**No-overwrite guard:** the stage wrapper refuses to run if `data/processed/geo200/` already contains `pilot_preprocessed.npz`.

| Kept identical | Check |
|---|---|
| Split boundaries [0, 6336, 7200, 8928] | `split_config.json` equals the primary run's file field by field |
| 12 → 12 windows; all targets inside a split; inputs only in the past | window-index checks |
| Training-only scaler and time-of-day medians; causal fill; NaN targets with mask | the same checks `scripts/verify_artifacts.py` applies to the primary run |
| Features (B1 adds time-of-day in the loader, not in preprocessing) | unchanged loader |

**Artifacts** (written by `prepare_pilot`; names unchanged so the existing loader works): in `data/processed/geo200/`:
- `split_config.json`
- `scaler_parameters.json`
- `pilot_preprocessed.npz`
- `window_index.npz`
- `preprocessing_report.md`

Plus `checkpoint_manifest.json`, written by the stage wrapper, containing:
- SHA-256 of every artifact and of the input manifest;
- raw-file fingerprints before and after;
- git commit and dirty-file list;
- Python and package versions;
- command line;
- UTC time;
- sensor and window counts per split and client;
- masked-target counts per split.

**Integrity:**
- The raw January file, `node_order.npy` and the metadata are fingerprinted before and after. `prepare_pilot` already checks this.
- All files in `data/processed/` outside `geo200/` and `results/` must be byte-identical before and after.
- February–December files are never opened.

**Expected counts** [estimate: frozen window rule × 200 sensors; exact counts after preprocessing]:

| Split | Windows (max) | Note |
|---|---:|---|
| Train | ≤ 1,262,600 | 6,313 per sensor |
| Val | ≤ 170,600 | 853 per sensor |
| Test | ≤ 343,400 | 1,717 per sensor; stored by the frozen pipeline, never loaded |

Windows with all targets missing are dropped, so counts may be slightly lower.

---

## 4. Experiment matrix (12 runs)

| # | Arm | Seeds | Runner | Config (proposed) | Partition |
|---|---|---|---|---|---|
| 1–3 | Centralised B1 | 0, 1, 2 | `experiments/run_phase6.py --run geo200` | `phase8_centralised_b1.json` | all 200 sensors |
| 4–6 | Local-only B1 | 0, 1, 2 | `experiments/run_phase7.py --run geo200` | `phase8_local_b1.json` | geographic (50 per client) |
| 7–9 | IID-like FedAvg B1 | 0, 1, 2 | `run_phase7.py --run geo200` | `phase8_fedavg_iid_b1.json` | seeded balanced sensor partition, 50 per client, partition seed 20261027 |
| 10–12 | Geographic FedAvg B1 | 0, 1, 2 | `run_phase7.py --run geo200` | `phase8_fedavg_geo_b1.json` | `proposed_client` (50 per client) |

**IID-like partition:**
- Built with the existing `federated.iid_partition`: a permutation from the partition seed, dealt round-robin.
- Each sensor and its scaler have exactly one owner, and windows are never split across clients.
- The partition JSON and its SHA-256 are saved with every run. Its type and flow composition is reported.

**Comparability**

| Item | Rule |
|---|---|
| Initial weights | `seed_everything(seed)` immediately before model construction in every arm. The same seed gives identical initial weights everywhere |
| Shuffle streams | centralised: `make_loader(seed)`; FedAvg: `client_seed(seed, round, k)`; local-only: `client_seed(seed, 0, k)` (as in Phase 7) |
| Budget | one centralised epoch = 1,262,600 sample-passes ≈ 2,467 steps; one FedAvg round = 4 clients × ≈ 315,650 = the same sample-passes ≈ 4 × 617 = 2,468 steps; local-only: 617 steps per client-epoch. Same maximum (50) and patience (5) |
| Evaluation | identical validation windows, masks and targets. Checked by hashing `(sensor_pos, t0, mask, y_true)` across all 12 runs |

**Unavoidable differences** (reported, not corrected):
- partial-batch rounding (≤ 4 steps per round);
- local-only makes 4 early-stopping decisions (one per client on its own validation), the others make 1;
- wall-clock is sequential simulation time.

### 4.1 Proposed configurations

Shared fields: identical to `experiments/configs/phase6_b1.json`.

```json
{ "arm": "centralised_b1_geo200", "run": "geo200", "calendar": true, "hidden_size": 64, "optimizer": "adam",
  "lr": 0.001, "batch_size": 512, "max_epochs": 50, "patience": 5, "grad_clip": 1.0, "lr_schedule": null,
  "early_stopping_metric": "validation pooled MAE, raw units", "clip_negative_predictions": false,
  "seeds": [0, 1, 2], "threads": 8, "results_subdir": "phase8" }

{ "arm": "local_b1_geo200", "partition": "geographic", "max_epochs": 50,
  "early_stopping_scope": "per_client_own_validation", ...shared..., "results_subdir": "phase8" }

{ "arm": "fedavg_iid_b1_geo200", "partition": "iid_sensor_balanced", "partition_seed": 20261027,
  "rounds_max": 50, "local_epochs": 1, "client_fraction": 1.0, "aggregation_weights": "train_windows",
  "aggregation_dtype": "float64", "optimizer_state": "reset_each_round", ...shared..., "results_subdir": "phase8" }

{ "arm": "fedavg_geo_b1_geo200", "partition": "geographic", ...same FedAvg fields..., "results_subdir": "phase8" }
```

`...shared...` = the B1 fields in the first config.

**No hyperparameter changes are proposed.** One open point: `threads` stays 8. The pilot timing was measured with 8 threads, and changing it would alter determinism, not results.

**Outputs:** `results/phase8/geo200/<arm>_seed<k>/`, which is git-ignored (`results/` is ignored). Each holds:
- `metrics.json` (with meta, history, summary and metrics);
- the exact config copy;
- `partition.json` (federated and local arms);
- best checkpoint(s);
- `predictions_<arm>_<seed>_val.npz`.

---

## 5. Evaluation and statistical analysis plan (pre-declared before any training)

### 5.1 Metrics (validation only; raw units unless stated; observed targets only; `n` always reported)
- **Pooled (micro) MAE and RMSE.** The primary endpoint is **pooled MAE**.
- **Normalised MAE** (z-units): secondary. It matters here because ramps have low flow and raw MAE is dominated by high-flow mainline sensors.
- **Sensor-macro MAE and RMSE.**
- **Per client:** micro and macro MAE/RMSE, for the 4 geographic regions in every arm (including the IID arm, where clients differ from regions).
- **Per horizon:** 5 … 60 min.
- **Per sensor:**
  - the MAE distribution (median, IQR, 5–95%);
  - per type group and per flow tertile;
  - the count of sensors on which each arm beats each other arm.
- **Negative-prediction rate** (not clipped).
- **Training:**
  - wall-clock seconds;
  - optimiser steps (total and to the best epoch or round);
  - skipped batches and clipped steps;
  - largest gradient norm.
- **Convergence:**
  - validation MAE per epoch or round;
  - best epoch or round;
  - early-stopping flag;
  - first round within 1% of the arm's best;
  - first round below the historical-average reference.
- **Reference forecasts** on the same validation windows: persistence, daily seasonal-naive, historical average.
- **Communication** (FedAvg): model-payload bytes per round and in total, and to the best round. Evaluation downloads reported **separately**.

### 5.2 Comparisons (all paired on identical targets)

| ID | Contrast | Question |
|---|---|---|
| C1 | geographic FedAvg − centralised | RQ1 |
| C2 | geographic FedAvg − local-only | RQ2 |
| C3 | IID FedAvg − geographic FedAvg | RQ3 |
| C4 | local-only − centralised | context |

Each contrast is reported on pooled MAE (primary), RMSE, normalised MAE, per client and per horizon.

### 5.3 Uncertainty and evidence
- **Seeds (3):** mean ± SD and per-seed paired differences. "Consistent across seeds" (all three the same sign) is **descriptive only and never called significant**. Seeds capture training randomness only, not sampling of sensors or time.
- **Sensor-cluster bootstrap (primary uncertainty):**
  - **Unit:** the sensor. All of a sensor's validation windows stay together, which absorbs the strong temporal autocorrelation and the 12-fold overlap of windows.
  - **Paired:** per sensor, compute each arm's seed-averaged absolute-error sum and cell count. Resample sensors **within each geographic client** (stratified), 2,000 replicates, fixed bootstrap seed 20261028. Recompute the pooled MAE difference each time.
  - **Report:** the 95% percentile CI and a two-sided bootstrap p-value (share of replicates on the other side of 0, × 2).
- **Multiplicity:** Holm adjustment across the four pre-declared contrasts C1–C4 on the primary endpoint. Everything else (RMSE, normalised MAE, per client, per horizon, per type) is **exploratory**: CIs shown, no claims.
- **Spatial-correlation sensitivity:** neighbouring sensors on the same freeway are correlated, which can make a sensor bootstrap too narrow. A second bootstrap resamples **freeway-direction clusters** (`Fwy` + `Direction`, provisional metadata) within each client. Where the two disagree, the wider interval is reported.
- **Temporal sensitivity:** validation is 3 consecutive weekdays (23–25 January) with no weekend. A **leave-one-day-out** analysis recomputes C1–C4 on each pair of days. With only 3 days this shows stability; it is not an interval.
- **Smallest effect size of interest (SESOI)** [proposed, D8-8]: a difference counts as **practically meaningful** only if its magnitude is ≥ 1% of the centralised pooled MAE. This is fixed after the centralised runs but **before** any other arm is analysed: centralised is analysed first and its value is written to the analysis log.
- **Wording rules:**

  | Situation | Wording |
  |---|---|
  | CI excludes 0, the Holm-adjusted p < 0.05, **and** \|Δ\| ≥ SESOI | "supported on January validation data" |
  | CI excludes 0 but \|Δ\| < SESOI | "detectable but below the practical threshold" |
  | Otherwise | "not distinguishable on this validation set" |

  No result is described as confirmed. Confirmation would need the closed test split in a separately approved phase, with frozen configs, run once.
- **RQ3 ("does geographic partitioning matter"):** answered by C3 under the wording rules above. It is reported together with the descriptive client-heterogeneity statistics (RQ5) and with per-client C3 differences.

### 5.4 Validation reuse
- Validation is used for early stopping in every run and for all Phase 8 comparisons. No hyperparameter, feature or partition choice may be made from Phase 8 validation results without a new, logged approval.

---

## 6. Reproducibility and safeguards

| Safeguard | Mechanism |
|---|---|
| Test split closed | `load_run(..., splits=("train", "val"))`; `run_phase7.py` has no test option; `run_phase6.py --evaluate test` needs `--final` plus a hash and is not used. Test windows are never built |
| No February–December access | no code path references `2023_p02` … `2023_p12`; the stage scripts assert the only raw file opened is `2023_p01.npy`; size and modification time of all 12 files are fingerprinted |
| Pilot untouched | `data/processed/` outside `geo200/`, `results/phase6`, `results/phase7`, `results/phase7_ablation` and the existing configs are fingerprinted before and after every stage |
| No leakage into selection or normalisation | selection uses training-period columns only (§2.1), enforced by an allow-list and a test. Scaler and medians are training-only (frozen pipeline) |
| No publication of sensitive artifacts | manifests, the selection report, preprocessing outputs and prediction records stay under git-ignored `data/` and `results/`. Docs contain aggregate counts only. No commit or push without approval (D-G still deferred) |
| Metadata per run | config copy plus SHA-256, partition plus SHA-256, seed, package versions, threads, `windows_loaded`, git commit, UTC time |
| Finite checks | the run stops on a non-finite loss or gradient norm (existing guard); checks on predictions and on weights and checkpoints |
| No overwrite | selection, preprocessing wrapper, `run_phase7.py` (existing) and `run_phase6.py` (to be added) refuse non-empty output folders, checking before data loading |
| Run completion | confirmed from files on disk (memory note: background notifications are unreliable on this machine) |

---

## 7. Resource budget

| Quantity | Value | Basis |
|---|---|---|
| Train / val windows | ≤ 1,262,600 / ≤ 170,600 | [estimate] frozen window rule × 200 |
| Steps per centralised epoch / FedAvg round | ≈ 2,467 / ≈ 2,468 | [estimate] |
| Time per step (B1, batch 512, 8 threads) | 7.6 ms centralised (3.75 s / 494 steps); ≈ 7.7 ms FedAvg (0.95 s / 124 steps per client epoch) | [measured] Phase 6/7 |
| Epoch / round time | ≈ 19–20 s (including about 0.7 s validation, measured at 0.13 s per 34k windows) | [estimate], assumes linear scaling |
| Run lengths | pilot: centralised 23–29 epochs, FedAvg 20–44 rounds, local-only 83–112 client-epochs | [measured] |
| Time per run | centralised ≈ 8–10 min; FedAvg ≈ 7–15 min; local-only ≈ 7–9 min. **Worst case** (50 epochs or rounds, 200 client-epochs) ≈ 16–17 min | [estimate] |
| Total, 12 runs | ≈ **1.5–2.5 h expected; ≤ 3.4 h worst case** (sequential, CPU) | [estimate] |
| Preprocessing | minutes (reads 200 rows of the memory-mapped array) | [estimate]; not measured at this size |
| Process memory | 0.42 GB observed during a 40-sensor local-only run (single observation); ≈ 0.5–0.6 GB expected (+25 MB arrays, +25 MB client copies, ≈ 20 MB predictions) | [measured] + [estimate] |
| Disk | preprocessing ≈ 8 MB npz + 3 MB index (pilot 1.6 + 0.6 MB for 40); per run ≈ 15 MB predictions + ≈ 0.2 MB metrics + checkpoints (58 KB each); 12 runs ≈ 0.2 GB | [estimate] scaled from [measured] |
| FedAvg communication | 59,413 B per model transfer; 475,304 B per round (unchanged: it depends on model size, not data); 20–44 rounds → 9.5–20.9 MB per run (≤ 23.8 MB at 50 rounds); evaluation downloads 4.8–10.5 MB (≤ 11.9 MB) separately | [measured] per round; totals [estimate] |
| Local-only and centralised communication | 0 model bytes | — |

---

## 8. Risks, limitations and mitigations

| Risk or limitation | Mitigation or reporting |
|---|---|
| Station-ID mapping provisional (A6), and with it `Type`, District and client labels | stated in every report; no claim of verified geography; strata labels described as provisional metadata |
| Type metadata may be wrong for some sensors | stratification is a design aid, not an analysis claim; per-type results are exploratory |
| Raw MAE dominated by high-flow mainline sensors | normalised MAE, per-type and per-sensor distributions reported |
| Short validation (3 weekdays) and temporal correlation | sensor-cluster bootstrap; leave-one-day-out; no weekend generalisation claimed |
| Spatial correlation between sensors | freeway-direction cluster bootstrap |
| Only 3 seeds | seeds used for training variability only; inference through the bootstrap |
| Validation reuse (B1 selection, early stopping, comparisons) | no new tuning on Phase 8 validation; test split reserved for a later, separately approved confirmation |
| Equal client sizes (no quantity skew) | stated; quantity-skew partitions remain FUTURE |
| Near-duplicate series not detected (only exact copies) | stated. Optional report-only check: training-period correlation ≥ 0.999 between selected pairs (D8-11) |
| Larger runs take longer (≤ 3.4 h) | sequential execution with fingerprints; completion verified on disk |
| Simulation, not deployment | stated (same as Phase 7) |
| January only | full-year scaling remains FUTURE and needs separate approval (11 months unopened) |
