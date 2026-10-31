# PS-003 Implementation Plan — Federated Traffic-Flow Prediction

**Project:** AIORI Hackathon 3, PS-003 — Federated Traffic-Flow Prediction with Communication-Cost Evaluation
**Workspace:** the project root (all commands run from here)
**Plan written:** 2026-10-24
**Handoff source:** `AIORI_Project_Context.md` (referred to in the task brief as `AIORI_context_complete.md`)

Status labels used below: **verified** (checked on this machine), **design choice** (decided here, can be revisited), **assumption** (believed but not proven).

---

## 0. Workspace state at plan time

| Item | Status | Notes |
|---|---|---|
| `year_2023/2023_p01.npy` | verified | `(16972, 8928, 3)`, `<f4`, C-order; opened with `mmap_mode="r"`. Lives in `year_2023/`, not the project root. |
| `year_2023/2023_p02.npy` … `p12.npy` | verified (headers only) | **All 12 months are present**, unlike the handoff ("January only"). Time lengths match days × 288 (e.g. Feb = 8064). Not used yet. |
| `node_order.npy` | verified | `(16972,)`, `int64`, all unique; same ID set as metadata `station_id`. |
| `sensor_meta_feature.csv` | verified | Tab-separated, 16,972 × 27. CSV row order ≠ `node_order` order (expected). District counts match handoff §7. 26 sensors lack Lat/Lng; 0 lack County. |
| `incidents_y2023.csv` | present | Out of scope for the core pipeline. |
| `rebuild_aiori_metadata.py` | verified, run 2026-10-24 | `ARRAY_PATH` fixed to `year_2023/2023_p01.npy`. Regenerated `data/processed/sensor_metadata_aligned.csv` and `proposed_client_assignment.csv`. Independently checked against every handoff §6–7 figure: client counts 4,857/4,072/2,974/5,069; district groups; per-district counts; missing geo per client 5/0/0/21; 0 mismatches across 26 shared columns. All PASS. |
| `.venv` | created 2026-10-24 | No Python existed on this machine. Installed Python 3.12.10 (user scope, winget); `.venv` with numpy 2.5.4, pandas 3.1.0, pytest 9.1.1. |
| Prior artifacts named in the handoff | **absent** | `check_clients.py`, `check_alignment.py`, `src/complete_inspection.py`, `reports/*.ipynb`, `data/processed/*`, `inspection_final.md` are not in this workspace. The prior PASS results cannot be re-run here until regenerated; they are treated as documented history, not local evidence. |

**Copied series:** 1,902 sensors carry byte-identical flow series and are excluded from the pilot. Details in `docs/data_quality_notes.md`.

**Row-to-ID mapping:** public `2023_p01.npy[i]` ↔ `node_order.npy[i]` remains **supported but unconfirmed** (handoff §8). Every artifact that carries a station ID, District or client derived via `node_order` marks it provisional.

---

## 1. Module structure

A flat package at the project root (no install step; run from the root). The existing metadata-rebuild script is reused rather than duplicated.

```
Traffic/
├── year_2023/*.npy, node_order.npy, sensor_meta_feature.csv   # raw, read-only
├── rebuild_aiori_metadata.py      # existing; reused (district→client map, aligned metadata)
├── aiori/
│   ├── __init__.py
│   ├── config.py                  # paths, channel indices, constants, seeds
│   ├── data_io.py                 # validated mmap loading, aligned metadata, raw-file guards
│   ├── flow_audit.py              # per-sensor Ch0 statistics, eligibility, pilot selection (+ CLI)
│   ├── splits.py                  # chronological boundaries → split_config.json        (Phase 4)
│   ├── preprocessing.py           # masks, training-only scaler, causal imputation        (Phase 3)
│   ├── windows.py                 # 12→12 sliding windows per split                      (Phase 5)
│   ├── prepare_pilot.py           # CLI: runs phases 3–5 for the pilot manifest           (Phases 3–5)
│   ├── model.py                   # compact GRU + masked loss                             (Phase 6)
│   ├── metrics.py                 # masked MAE/RMSE with observed-target counts           (Phase 6)
│   ├── partitions.py              # IID-like and geographic client membership             (Phases 9–10)
│   ├── training.py                # shared train/eval loop (centralized + local-only)     (Phases 7–8)
│   ├── fedavg.py                  # client update, weighted aggregation, byte accounting  (Phases 9–11)
│   └── reporting.py               # tables + figures from saved logs                      (Phase 12)
├── experiments/
│   ├── configs/pilot.json         # one config for all arms
│   └── run.py                     # --arm {centralized,local,fedavg_iid,fedavg_geo} --seed N
├── tests/                         # synthetic-array unit tests; no full-dataset loads
├── data/processed/                # all generated data artifacts (never raw)
├── results/                       # per-run logs, metrics, checkpoints, figures
└── docs/implementation_plan.md
```

Dependencies: numpy, pandas, pytest now; **PyTorch** added in Phase 6; **matplotlib** in Phase 12. Nothing else.

---

## 2. Phases

### Phase 1 — Workspace and data validation
- **Objective:** Fail fast on missing files, unexpected shapes and dtypes, and duplicate or mismatched IDs; regenerate the derived metadata that is missing on this machine.
- **Files:** `aiori/config.py`, `aiori/data_io.py` (today); `rebuild_aiori_metadata.py` (one-line `ARRAY_PATH` fix to `year_2023/2023_p01.npy`, only with approval, since it is an existing file).
- **Inputs → outputs:** raw files → validated memmap + aligned metadata frame; `data/processed/sensor_metadata_aligned.csv`, `proposed_client_assignment.csv` (from the rebuild script).
- **Dependencies:** none.
- **Tests / acceptance:** shape/dtype/ndim guards raise on bad synthetic arrays; duplicate-ID and ID-set mismatch raise; aligned metadata `station_id` equals `node_order` in order; regenerated assignment reproduces the handoff counts 4,857 / 4,072 / 2,974 / 5,069 and district groups `[7] [4] [5,12] [3,6,8,10]`.
- **Assumptions:** row-to-ID mapping as above.

### Phase 2 — Sensor-quality analysis and pilot selection *(started today)*
- **Objective:** A reproducible Channel 0 (flow) audit for all 16,972 sensors and a 20–50 sensor pilot.
- **Files:** `aiori/flow_audit.py`, `tests/test_flow_audit.py`.
- **Inputs → outputs:** `2023_p01.npy` Ch0 (memmap, batched) + aligned metadata → `data/processed/flow_sensor_audit.csv` (all sensors), `data/processed/sensor_selection.csv` (pilot), `data/processed/flow_quality_report.md`.
- **Dependencies:** Phase 1 loaders.
- **Tests / acceptance:** correct finite/missing counts on synthetic data; zeros counted as observed; fully missing and constant flags; deterministic selection; unique IDs; raw files unchanged (size + mtime) after a run.
- **Assumptions:** Ch0 = flow (working convention from the upstream notebook, not independently confirmed). Selection stratified by the provisional geographic client so that every partition has data; this does not use any forecasting performance.
- **Revision 2 (2026-10-24, approved review changes):**
  - Primary eligibility now uses the **training period only** (steps [0, 6336)); whole-month statistics are informational `month_*` columns.
  - Exact-duplicate exclusion is kept as a separate, precautionary rule on training-period hashes: 1,944 sensors in 104 groups, 42 of them copies only during training.
  - A **District floor** guarantees every District in every client: coastal = District 5 ×1 + District 12 ×9. The pilot is still 40 sensors, 39 of them the same as revision 1.
  - A separate **8-sensor missing-data validation set** (`missing_validation_selection.csv`, 99–<100% January coverage, val/test gaps) exercises fill and masking on real data.
  - Revision 1 outputs are archived in `data/processed/archive/v1_month_eligibility/`. Eligible sensors: 11,188 (was 11,198).
  - Verified by `scripts/verify_artifacts.py`: 85/85 PASS. Details in `docs/preprocessing_review.md`.
- **Status (2026-10-24, revision 1):** done for January. All handoff Channel 0 counts reproduced exactly. **New finding:** 1,902 fully observed sensors share byte-identical series with another sensor (97 groups, largest 449, none spanning Districts), which points to upstream fill. They are excluded via `duplicate_series`. Pilot = 40 sensors (10 per client). Within the coastal client, all 10 sensors are District 12; none come from District 5.

### Phase 3 — Missing-value masks and leak-safe preprocessing
- **Objective:** Turn the pilot series into model inputs without leakage.
- **Files:** `aiori/preprocessing.py`, `tests/test_preprocessing.py`.
- **Inputs → outputs:** pilot `(S, T)` flow slice + split boundaries → observed mask `(S, T)`; scaler fitted on **train-period observed values only** (`data/processed/scaler_parameters.json`; per-sensor or global — decide before Phase 7 and record it); scaled inputs with masked-out positions.
- **Input imputation (design choice from audit):** linear interpolation for input gaps ≤ 6 steps (30 min), computed **within one split segment only**; longer gaps filled with a train-period per-sensor time-of-day median. Targets are **never** imputed and are masked in the loss and metrics. An input-was-imputed flag is kept so it can be used as a feature or ablation.
- **Dependencies:** Phase 4 boundaries (the scaler needs the train period).
- **Tests / acceptance:** zeros survive; NaN targets stay masked; the scaler is unchanged when validation/test values are perturbed; no interpolation crosses a boundary; inverse transform round-trips.
- **Status (2026-10-24): done** in `aiori/preprocessing.py`. Decisions:
  - **Scaler = per-sensor z-score** (design choice). Each client can fit its own sensors locally without pooling data across clients; pilot flow levels range 13–404. `--scaler-mode global` remains available.
  - **Imputation changed from the audit proposal:** linear interpolation was rejected because its right endpoint can be a future (target) value. Inputs now use a causal fill: forward fill ≤ 6 steps without crossing split boundaries, then train time-of-day median, then train mean. An observed/imputed flag is an input feature.
  - The current pilot is 100% observed, so no imputation is triggered on it. Unit tests cover the logic.

### Phase 4 — Chronological train/validation/test split
- **Objective:** Fixed timestamp boundaries shared by all arms and seeds.
- **Files:** `aiori/splits.py`, `data/processed/split_config.json`, `tests/test_splits.py`.
- **Default (January pilot):** 70/10/20 by timestamp on 8,928 steps, boundaries snapped to whole days: train 2023-01-01 → 01-22 (22 d), val 01-23 → 01-25 (3 d), test 01-26 → 01-31 (6 d). Timestamps come from the upstream notebook convention (`2023-01-01 00:00`, 5 min, no proven timezone).
- **Dependencies:** none.
- **Tests / acceptance:** boundaries are monotonic and cover the full range with no overlap; the JSON round-trips.
- **Status (2026-10-24): done.** `data/processed/split_config.json`: train [0, 6336) Sun 1 – Sun 22 Jan; val [6336, 7200) Mon 23 – Wed 25 Jan; test [7200, 8928) Thu 26 – Tue 31 Jan.
- **Open decision (new information):** all 12 months exist locally. A month-level split (e.g. train Jan–Sep, val Oct, test Nov–Dec) would support stronger claims, at higher I/O cost. Recommend finishing the January pilot end-to-end first, then deciding.

### Phase 5 — Sliding-window generation
- **Objective:** 12 input → 12 target windows whose **targets** fall entirely inside one split; input context may reach back into the previous split.
- **Files:** `aiori/windows.py`, `tests/test_windows.py`.
- **Outputs:** per split, `X (N, 12, F)`, `Y (N, 12)`, `Ymask (N, 12)`, plus `(sensor, t0)` index. Windows are generated lazily (index-based dataset) rather than materialising every window on disk.
- **Policy:** drop windows with zero observed targets; keep partially observed windows (masked).
- **Tests / acceptance:** counts match closed-form expectations; no target index crosses a boundary; the mask agrees with the source NaNs.
- **Status (2026-10-24): done.** `aiori/windows.py` + `aiori/prepare_pilot.py` → `window_index.npz` (`[sensor_position, t0]`), `pilot_preprocessed.npz`, `scaler_parameters.json`, `preprocessing_report.md`. Pilot windows: train 252,520 / val 34,120 / test 68,680 (= 6,313 / 853 / 1,717 per sensor, matching closed form). Inputs are `[scaled flow, observed flag]`; time features deferred.

### Phase 6 — Compact GRU and masked loss
- **Objective:** One small architecture for every arm.
- **Files:** `aiori/model.py`, `aiori/metrics.py`, `tests/test_model.py`, `tests/test_metrics.py`. Adds PyTorch (CPU build is sufficient for the pilot).
- **Design:** GRU(input F → hidden 64, 1 layer) → Linear(64 → 12). Masked MSE loss; MAE/RMSE reported in original units with the count of observed targets.
- **Tests / acceptance:** output shapes; a loss with all-masked targets is handled without NaN; masked metrics match hand-computed values; can overfit a tiny batch; parameter count logged.

### Phase 7 — Centralized baseline
- **Files:** `aiori/training.py`, `experiments/run.py`, `experiments/configs/pilot.json`.
- **Behaviour:** pool all pilot sensors' training windows; early stopping on validation MAE; evaluate once on test.
- **Outputs:** `results/<arm>/<seed>/` with config, checkpoint, `metrics_per_epoch.csv`, `test_metrics.csv`.
- **Acceptance:** finite losses; the run is reproducible for a fixed seed; it beats a naive last-value / historical-average baseline (which should also be reported).

### Phase 8 — Local-only baseline
- **Behaviour:** one independent GRU per client on that client's sensors only, with the same training loop and budget; evaluate each on its own test sensors and report an observed-target-weighted overall score.
- **Acceptance:** the client sensor sets are disjoint and their union equals the pilot set.
- **Open:** which partition defines "local" (geographic, IID, or both). Recommend reporting local-only under **both** partitions so each FedAvg arm has a matching local comparison.

### Phase 9 — IID-like FedAvg
- **Files:** `aiori/partitions.py`, `aiori/fedavg.py`, `tests/test_partitions.py`, `tests/test_fedavg.py`.
- **Behaviour:** seeded random assignment of pilot sensors to 4 clients, with the same client sizes as the geographic split; full participation; E = 1 local epoch per round; weighted average with `n_k` = local training windows.
- **Outputs:** `data/processed/client_membership.csv` (strategy, seed, array_index, client).
- **Tests / acceptance:** aggregation weights sum to 1; averaging identical models is a no-op; a hand-computed 2-client average matches; all clients start each round from identical weights.

### Phase 10 — Geographic non-IID FedAvg
- **Behaviour:** clients = provisional District groups `[7] [4] [5,12] [3,6,8,10]` from `CLIENT_BY_DISTRICT` in `rebuild_aiori_metadata.py`.
- **Acceptance:** same tests as Phase 9; client membership written with the mapping caveat.
- **Assumption:** geographic interpretation depends on the unconfirmed row-to-ID mapping and is stated in every result.

### Phase 11 — Communication bytes and convergence logging
- **Behaviour:** serialise the state dict (`torch.save` to a `BytesIO`) for each downlink and each uplink and record `len(bytes)`; also log the theoretical `params × 4 × clients × 2`. Per round: validation MAE/RMSE, uplink, downlink and cumulative bytes, wall time.
- **Outputs:** `results/<arm>/<seed>/communication.csv`, `metrics_per_round.csv`.
- **Tests / acceptance:** measured bytes ≥ raw parameter bytes and constant across rounds; cumulative bytes equal the running sum. The error threshold for "rounds to target" is fixed **before** looking at test results (e.g. within 5% of centralized validation MAE).

### Phase 12 — Three-seed experiments, evaluation, plots, report
- **Behaviour:** seeds `[0, 1, 2]` (design choice, fixed in config) × 4 arms (local-only for both partitions); identical test period; mean ± std.
- **Files:** `aiori/reporting.py`, `results/summary.csv`, `results/figures/` (convergence, bytes vs. validation error, IID vs. non-IID bars).
- **Acceptance:** every reported number traces to a saved log; limitations section covers the January-only pilot (or the chosen months), the mapping caveat, simulation-only federation, no formal privacy, and the CC BY-NC 4.0 licence.

---

## 3. Reproduce and verify (current state)

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m aiori.flow_audit
.venv/Scripts/python.exe -m aiori.prepare_pilot
.venv/Scripts/python.exe -m aiori.prepare_pilot --selection data/processed/missing_validation_selection.csv --out-dir data/processed/missing_validation --min-sensors 4 --run-label "missing-data validation set (not the primary pilot)"
.venv/Scripts/python.exe scripts/verify_artifacts.py
```

The raw-hash baseline (`data/raw_integrity_baseline.json`) is local, recorded 2026-10-24 after download; it is not the original-download hash.

## 4. Cross-cutting rules
- Raw files are opened read-only (`mmap_mode="r"`); outputs go only to `data/processed/` and `results/`, and code refuses to write to a raw path.
- Zero flow is a valid observation; NaN is missing; missing targets are never filled.
- No statistic may use validation or test observations; no interpolation across split boundaries.
- Sensors and missing values are never silently dropped; every exclusion has a recorded reason.
- Test metrics are never used for selection or tuning.

## 5. Decisions needed from the team
1. ~~Fix `ARRAY_PATH` in `rebuild_aiori_metadata.py`~~ — done 2026-10-24; outputs verified.
2. Keep January-only for the pilot, or design the final split around the 12 available months?
3. ~~Global vs per-sensor scaler~~ — per-sensor chosen 2026-10-24 (see Phase 3); revisit only with a reason.
4. Report local-only under both partitions (recommended) or only the geographic one?
5. ~~Duplicate series policy~~ — keep excluding all members (precautionary), decided 2026-10-24. Still open: run the check on the other 11 months before scaling to 200–300 sensors.
6. Resolved 2026-10-24 in review revision 2: training-period eligibility; District floor; separate missing-data validation set; keep the 22/3/6 split with the weekday-only-validation limitation documented; `scripts/verify_artifacts.py`.
7. ~~Open judgement calls before Phase 6~~ — decided 2026-10-24: keep excluding the 42 training-only duplicates; accept the current 8-sensor validation set (the empty-window drop and the mean fallback stay covered by unit tests only). **The data pipeline is frozen for Phase 6:** no further data changes.
