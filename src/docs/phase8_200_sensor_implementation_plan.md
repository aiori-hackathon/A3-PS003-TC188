# Phase 8 implementation plan — ≈ 200-sensor geographic experiment

**Date:** 2026-10-27. **Status (2026-10-28): S1–S5 complete (§8–§12); results in `docs/phase8_results.md`.** Implements `docs/phase8_200_sensor_technical_spec.md`.

**Boundaries until each gate below is approved:**
- no sensor selection or export;
- no preprocessing artifacts;
- no code changes;
- no smoke tests or training;
- no test-split access;
- no February–December files opened;
- no commit or push.

---

## 1. Stages and approval gates

| Stage | Work | Produces | Gate (needs explicit approval) |
|---|---|---|---|
| **S0** | Approve the design and decisions D8-1 to D8-12 (§5) | — | **G0: approve the plan** |
| **S1** | Implement the selection and checkpoint tooling with **synthetic tests only**: `aiori/select_geo200.py`, `scripts/prepare_geo200.py` (wrapper), `scripts/verify_geo200.py`, additive `dataset.RUNS["geo200"]`, `run_phase6.py` overwrite guard and config-driven output root, Phase 8 configs, analysis module. No real data is read except the existing test suite's real-artifact tests | code + tests; all existing tests still pass | **G1: approve generating the selection and checkpoint** |
| **S2** | Run the selection on the frozen audit (training-period columns only), then `prepare_pilot` into `data/processed/geo200/`; verify; fingerprint | manifest, summary, checkpoint, `checkpoint_manifest.json`, verification results; **aggregate** report to the lead | **G2: approve the selection and checkpoint** |
| **S3** | Tiny real-data smoke tests on `geo200`: each of the 4 arms, seed 0, 2 epochs/rounds × 5 batches per client; validation only; outputs in `results/phase8_smoke/` | smoke report (finite checks, counts, bytes, timing per step) | **G3: approve the 12 full runs** |
| **S4** | 12 runs (§3) sequentially; fingerprints before and after; completion verified on disk | `results/phase8/geo200/*` | — |
| **S5** | Pre-declared analysis (spec §5): centralised first (fixes the SESOI), then the other arms; bootstraps; report | `docs/phase8_results.md` (aggregate only) | **G5: decide the next phase** (for example a test-split confirmation of frozen configs, or full-year scaling; each separate) |

## 2. Code changes proposed for S1 (none made yet)

| File | Change | Kind |
|---|---|---|
| `aiori/select_geo200.py` | `stratified_selection(audit, meta, exclude, per_client=50, salt="aiori-geo200-v1")`; type-group map; within-(client, type) flow tertiles; largest-remainder allocation; hash ordering; fallbacks; District check and floor; **column allow-list** (raises if `month_*` or `eval_*` are passed in); input-hash check; refuses a non-empty output folder | new module |
| `scripts/prepare_geo200.py` | runs the selection then `aiori.prepare_pilot.main([...])` with the spec §3 arguments. Writes `checkpoint_manifest.json` (artifact hashes, raw fingerprints before/after, versions, git state, counts, masked-target counts per split). Asserts the only raw array opened is `2023_p01.npy` | new script |
| `scripts/verify_geo200.py` | read-only checks, reusing `scripts/verify_artifacts.py` check functions where they take a run directory: shapes, scaler fit range, mask equals NaN, window bounds, manifest order, split equality with the primary run, client counts (50 each), uniqueness, no overlap with the 48 pilot sensors | new script |
| `aiori/dataset.py` | add `RUNS["geo200"] = (PROCESSED_DIR / "geo200", PROCESSED_DIR / "geo200" / "sensor_selection.csv")` | additive; existing runs unchanged |
| `experiments/run_phase6.py` | (a) refuse a non-empty output folder, checked before loading; (b) default output root from config `results_subdir` (falls back to `phase6`, so existing configs are unchanged) | small change; Phase 6 behaviour preserved |
| `experiments/configs/phase8_*.json` | 4 configs (spec §4.1) | new |
| `aiori/phase8_analysis.py` (or `scripts/analysis/phase8_compare.py`) | paired contrasts; sensor-cluster and freeway-cluster stratified bootstraps; Holm; leave-one-day-out; SESOI logic; summary tables | new |
| Tests | `tests/test_select_geo200.py`, `tests/test_phase8_analysis.py`, plus additions for the runner guard and `RUNS` entry | new |

`aiori/preprocessing.py`, `windows.py`, `splits.py`, `flow_audit.py`, `prepare_pilot.py`, `model.py`, `losses.py`, `training.py`, `federated.py` and `evaluation.py` are **not modified**.

## 3. Experiment matrix and commands (S4; proposed)

```bash
# centralised (3)
.venv/Scripts/python.exe experiments/run_phase6.py --config experiments/configs/phase8_centralised_b1.json --run geo200 --seed {0,1,2} --baselines
# local-only, IID FedAvg, geographic FedAvg (9)
.venv/Scripts/python.exe experiments/run_phase7.py --config experiments/configs/phase8_local_b1.json      --run geo200 --seed {0,1,2}
.venv/Scripts/python.exe experiments/run_phase7.py --config experiments/configs/phase8_fedavg_iid_b1.json --run geo200 --seed {0,1,2}
.venv/Scripts/python.exe experiments/run_phase7.py --config experiments/configs/phase8_fedavg_geo_b1.json --run geo200 --seed {0,1,2}
```

| Run | Arm | Seed | Output folder (`results/phase8/geo200/`) |
|---|---|---|---|
| 1–3 | centralised | 0, 1, 2 | `centralised_b1_geo200_seed{k}` |
| 4–6 | local-only | 0, 1, 2 | `local_b1_geo200_seed{k}` |
| 7–9 | IID FedAvg | 0, 1, 2 | `fedavg_iid_b1_geo200_seed{k}` |
| 10–12 | geographic FedAvg | 0, 1, 2 | `fedavg_geo_b1_geo200_seed{k}` |

**Execution:**
- Order: centralised first (it fixes the SESOI), then local-only, IID, geographic.
- Sequential; one detached batch with a status file; completion confirmed from output files.
- Expected about 1.5–2.5 h, worst case about 3.4 h (spec §7).

## 4. Test and integrity checklist

### 4.1 Selection (S1 synthetic, S2 real)
- [ ] Only allow-listed training-period columns are used. Permuting `month_*` or `eval_missing_count` leaves the selection unchanged (test).
- [ ] Deterministic: the same inputs and salt give an identical manifest (test, and SHA-256 at S2). A different salt gives a different draw.
- [ ] 50 per client; 200 in total; unique `array_index`; one client per sensor; no overlap with the 48 pilot sensors.
- [ ] Allocation equals the largest-remainder proportions. Tertile splits follow the declared remainder order. Fallbacks are triggered and logged correctly on synthetic short strata.
- [ ] Every candidate is eligible (A1/A2); no exact-copy group member is selected.
- [ ] District check: each District with candidates in a client has ≥ 1 selected sensor, or the floor fallback is logged.
- [ ] Input-hash check fails closed on a changed audit file. The output folder must not exist or must be empty.

### 4.2 Checkpoint (S2)
- [ ] `prepare_pilot` raw-file fingerprints are unchanged. The size and modification time of all 12 monthly files are unchanged, and only `2023_p01.npy` is opened.
- [ ] Split boundaries are identical to the primary run.
- [ ] Scaler `fit_range` is [0, 6336]; inverse round trip is exact; `observed_mask` equals finite targets; targets are NaN exactly where missing.
- [ ] Window targets are inside their split; inputs start ≥ 0; window counts are reported per split and client.
- [ ] Manifest order equals npz order; `station_id` equals `node_order[array_index]` (consistency only, not verification).
- [ ] Counts of masked validation targets and imputed inputs are reported (aggregate).
- [ ] `checkpoint_manifest.json` is written. All pre-existing files under `data/processed/` and `results/` are byte-identical.

### 4.3 Training and evaluation (S3 smoke, S4 full)
- [ ] All existing tests pass (144 now) plus the new tests.
- [ ] Loader on `geo200`: shapes; finite `x` and `y`; mask equals the original NaNs; split isolation; client filter; test windows not loaded.
- [ ] Partition integrity: 4 × 50, disjoint and covering; the IID partition deterministic and different from the geographic one; partition SHA-256 saved.
- [ ] Client isolation (existing Phase 7 tests): sliced client copies, server type checks, no optimiser state transmitted.
- [ ] FedAvg aggregation equals the weighted mean (existing tests). `n_k` is recorded per client.
- [ ] Finite checks: losses and gradient norms (guarded in the loop), aggregated weights, predictions and checkpoints.
- [ ] Metric consistency: pooled metrics from client sums equal metrics recomputed from the merged records. The analysis module's pooled MAE equals `evaluation.score` on the same records.
- [ ] Identical validation targets across all 12 runs (hash of `sensor_pos`, `t0`, mask and `y_true`).
- [ ] No overwrite: every runner refuses non-empty run folders.
- [ ] Fingerprints before and after S3 and S4 (data, raw files, all existing results, code, configs).

### 4.4 Analysis (S5)
- [ ] Bootstrap is reproducible (fixed seed 20261028); on synthetic data with a known difference the CI covers it; it reduces to the pooled MAE difference at the point estimate.
- [ ] Holm adjustment is correct on known p-values.
- [ ] SESOI is recorded from the centralised runs before the other arms are analysed (analysis log with time).
- [ ] Reports contain aggregate statistics only: no station IDs and no raw observations.

## 5. Decisions requiring approval

| # | Decision | Recommendation |
|---|---|---|
| D8-1 | Exclude the 40 pilot and 8 missing-data-validation sensors from the candidate pool | approve (independent sample; pilot preserved as a separate testbed) |
| D8-2 | Strata: type groups mainline / hov / ramp (On+Off) / other (Fwy-Fwy, Coll/Dist, Conventional Highway), × training-mean-flow tertiles **within (client, type group)**; proportional largest-remainder allocation across types; equal tertile thirds (remainder mid → high → low) | approve |
| D8-3 | Within-stratum order by `sha256("aiori-geo200-v1:<array_index>")` (the documented seed); **no** ranking by missing count | approve |
| D8-4 | District check, with the A3-style floor fallback only if a District gets 0 | approve |
| D8-5 | Locations: `data/processed/geo200/` (checkpoint), `results/phase8_smoke/` (S3), `results/phase8/geo200/` (S4), `docs/phase8_results.md` (aggregate report) | approve |
| D8-6 | S1 code changes in §2: new modules and scripts; additive `RUNS` entry; `run_phase6.py` overwrite guard and output root | approve |
| D8-7 | Hyperparameters and FedAvg settings identical to the pilot (B1, reset_each_round, IID partition seed 20261027, 8 threads) | approve |
| D8-8 | Primary endpoint pooled MAE. Sensor-cluster stratified bootstrap (2,000 reps, seed 20261028) as the primary uncertainty, plus a freeway-direction cluster sensitivity, leave-one-day-out, and Holm over C1–C4. SESOI = 1% of centralised pooled MAE, fixed before analysing the other arms | approve or adjust the SESOI |
| D8-9 | Wording rules (spec §5.3); no significance from seed agreement; no "confirmed" without a separately approved test phase | approve |
| D8-10 | Descriptive client-heterogeneity statistics (R2 features, η², pseudo-F) on the selected 200 sensors, training period only, run at S2 | approve (optional) |
| D8-11 | Optional report-only near-duplicate screen: training-period correlation ≥ 0.999 among selected pairs, reported without changing the selection | your call |
| D8-12 | Commit policy: everything remains uncommitted (D-G still deferred). Manifests and reports with sensor IDs never committed | confirm |

## 6. Missing information (not guessed)
1. **Correctness of `Type`, District and Fwy metadata** per sensor: inherited from the provisional `node_order` join; not verifiable in-project.
2. **Validation/test missingness of the eventual 200:** unknown until S2 by design. It may produce more masked targets than the pilot (22.6% of candidates have training gaps).
3. **Scaling of per-step time from 40 to 200 sensors:** assumed linear in window count (per-step cost is unchanged because the batch size is fixed). Measured at S3.
4. **`prepare_pilot` runtime for 200 sensors:** not measured; expected minutes.
5. **Whether the R2 heterogeneity estimate (η² ≈ 0.10) holds for a type-stratified sample:** answered descriptively at S2 (D8-10).
6. **A power analysis for forecast-error differences between arms:** none exists. Phase 8 reports intervals rather than promising detection.

## 7. Next stage requiring approval
**G0:** approve this plan and decisions D8-1 to D8-12. After that, **S1** (code and synthetic tests only) can start. Generating the actual selection and checkpoint (S2) needs **G1**.

## 8. G0 record and S1 status (2026-10-28)

**Decisions at G0:**
- D8-1 to D8-7 and D8-9: approved as recommended.
- D8-8: the analysis module and the SESOI are **deferred** to a separate approval before S5. They are not part of S1.
- D8-10: **deferred**; decide at G1/G2.
- D8-11: **included**, report-only, in `scripts/verify_geo200.py`. Pairs with training-period r ≥ 0.999 are reported and the selection is never changed.
- D8-12 (replaces "everything uncommitted"): S1 code, tests, configs and aggregate docs may be committed **with approval**. Manifests, selection reports, checkpoints and results are never committed.

**S1 files:**

| File | Kind |
|---|---|
| `aiori/select_geo200.py` | new: allow-listed selection, allocation, hash draw, fallbacks, District floor, input-hash check, no-overwrite outputs, config loader, near-duplicate screen |
| `scripts/prepare_geo200.py` | new: S2 wrapper (selection → unchanged `prepare_pilot` → `checkpoint_manifest.json`) |
| `scripts/verify_geo200.py` | new: read-only S2 checks (reuses `verify_artifacts.py`) |
| `experiments/configs/phase8_selection.json` | new: salt, 50 per client, inputs and their approved SHA-256, `prepare_pilot` bounds |
| `experiments/configs/phase8_{centralised,local,fedavg_iid,fedavg_geo}_b1.json` | new: pilot settings plus `run`, `derived_from` and `results_subdir: phase8` |
| `aiori/dataset.py` | additive `RUNS["geo200"]` |
| `experiments/run_phase6.py` | output root from `results_subdir` (default `phase6`); refuses a non-empty run folder before loading |
| `tests/test_select_geo200.py`, `tests/test_phase8_configs.py` | new synthetic tests |

**Implementation notes:**
- `duplicate_series` (training-period) is on the allow-list as a guard only: an eligible exact copy raises.
- An unmapped or missing `Type` among the candidates fails closed.
- Tertiles for type groups with fewer than 3 candidates: 1 → mid; 2 → low, high (qcut-consistent).
- District floor: "largest stratum" means most drawn sensors, ties by name. Only a sensor whose District keeps ≥ 1 is replaced. If none exists the run fails closed and does not extend the rule.
- The checkpoint manifest counts test windows but reads no test targets. Masked-target counts are for train and val only.

**Next gate: G1** approves running `scripts/prepare_geo200.py` and `scripts/verify_geo200.py` on the real frozen inputs.

## 9. S2 status (2026-10-28, aggregate only)
- Selection: 200 sensors, 50 per client, from 11,140 candidates (48 pilot sensors excluded); 0 fallbacks; District floor not needed.
- Type groups: mainline 128, ramp 41, HOV 26, other 5. Per client, mainline/ramp/HOV/other: LA/Ventura 27/13/9/1, Bay Area 36/13/0/1, Coastal/South Central 30/7/11/2, Inland/Central 35/8/6/1.
- Training coverage: 46 of 200 sensors have at least one missing training step (560 steps in total; minimum coverage 99.46%).
- Checkpoint `data/processed/geo200/` (local, git-ignored, about 10.5 MB): 1,262,580 train and 170,600 val windows; 6,480 masked train target cells and 24 masked val target cells.
- `verify_geo200.py`: 48 checks passed, 0 failed. Near-duplicate screen: 0 pairs with r ≥ 0.999.
- Integrity: all 276 pre-S2 fingerprints are unchanged. Only `2023_p01.npy` was opened.
- **Next gate: G3** (S3 smoke tests).

## 10. S3 status (2026-10-28, smoke only, validation only)
- All 4 arms ran at seed 0 with the smoke budget: 2 epochs or rounds × 5 batches (per client for the federated and local arms). The centralised arm ran with `--max-batches 5` to match this plan. Outputs are in `results/phase8_smoke/` (git-ignored).
- Every arm passed:
  - finite losses, gradient norms, predictions, metrics and checkpoints;
  - validation masks match the missing targets (24 masked cells);
  - identical validation targets across arms;
  - 0 skipped batches.
- The federated partitions are 4 × 50, disjoint and cover all 200 sensors. The geographic partition equals the manifest clients, and the IID partition differs. Pooled metrics computed from client sums equal the merged records. Each FedAvg round transfers 475,304 B.
- Rerunning each arm reproduced its predictions, checkpoints and history exactly.
- Peak memory was 550–580 MB per arm. Smoke time per optimiser step was about 10 ms for the federated and local arms (17.6 ms centralised, including warm-up), against 7.6 ms in the pilot.
- 184 tests pass. All 286 pre-S3 fingerprints are unchanged.
- **Next gate: G3** (the 12 full runs).

## 11. S4 status (2026-10-28, validation only, descriptive only)
- 12 of 12 runs completed sequentially (07:16–09:20 UTC, about 2 h 03 min). No run failed and none was repeated. Outputs are in `results/phase8/geo200/`, with logs and status in `results/phase8/logs/` (git-ignored).
- Every run passed:
  - finite losses, gradient norms, predictions, metrics and checkpoints;
  - masks match the observed targets;
  - identical validation targets across the 12 runs;
  - 0 skipped batches and 0 clipped steps;
  - partitions 4 × 50; pooled metrics from client sums equal the records; 475,304 B per round.
- Pooled validation MAE, mean ± SD over seeds 0–2: centralised 16.383 ± 0.098; local-only 16.458 ± 0.147; IID FedAvg 16.477 ± 0.087; geographic FedAvg 16.495 ± 0.098.
- Reference forecasts on the same windows: historical average 22.543, persistence 22.671.
- Geographic FedAvg seed 0 reached the 50-round cap without early stopping (best round 46).
- **No statistical analysis has been done.** The S5 analysis module and SESOI need separate approval.
- 184 tests pass. All 330 pre-S4 fingerprints are unchanged.

## 12. S5 status (2026-10-28)
- SESOI fixed at 0.1638 (1% of 16.383) and recorded before any contrast was computed (`results/phase8/analysis/sesoi_log.json`).
- Analysis code: `aiori/phase8_analysis.py` and `scripts/phase8_analysis.py`, with 12 synthetic tests (196 tests in total pass).
- Results are in `docs/phase8_results.md`, aggregate only:
  - C1 (geographic FedAvg − centralised) +0.112: detectable but below the practical threshold;
  - C2, C3 and C4: not distinguishable on this validation set.
- No test split was used and no model was retrained. All earlier artifacts are unchanged.
