# Phase 6 technical specification — compact GRU forecasting baseline

**Date:** 2026-10-27. **Status: PROPOSED.**

Nothing here is implemented. Items marked **[brief]** were specified by the project lead in the 2026-10-27 design brief. Items marked **[proposed]** are recommendations awaiting approval.

**Inputs:** the frozen preprocessing checkpoint (`f45bc9a`; artifacts in `data/processed/`). Preprocessing, splits and selection are unchanged.

**Research context:** `docs/research_validity_decisions.md`. Row-to-ID mapping is **supported by strong indirect evidence and unconfirmed by provenance**. The 40-sensor pilot validates the pipeline; it is not a basis for geographic non-IID conclusions.

---

## 1. Architecture

| Item | Specification | Status |
|---|---|---|
| Framework | PyTorch, CPU build | [brief]; install needs approval |
| Model | `ForecastGRU`: `nn.GRU(input_size=F, hidden_size=64, num_layers=1, batch_first=True)` → last hidden state `h_12` (B,64) → `nn.Linear(64, 12)` | [brief] for GRU / 1 layer / 64; head [proposed] |
| Input | `X` float32 **(B, 12, F)**, steps t0−12 … t0−1 | [brief] |
| Output | `Ŷ_z` float32 **(B, 12)**: z-scaled flow at t0 … t0+11 (5 … 60 min ahead) | [brief] |
| Features, baseline B0 (F = 2) | `[0]` z-scaled flow after causal fill; `[1]` observed flag: **1.0 = raw value observed, 0.0 = value was missing and filled** | [brief] |
| Features, variant B1 (F = 4) | B0 + `sin(2π·slot/288)`, `cos(2π·slot/288)` with slot = t mod 288 for each input step | [proposed] optional |
| Day-of-week feature | **excluded** | [proposed] |
| Sensor ID / embeddings | **excluded** | [proposed] |
| Normalisation layers | none (no BatchNorm or running statistics) | [proposed] |

**Why time-of-day is allowed and day-of-week is not (training data only, measured 2026-10-27):**
- **Time of day:** the reconstructed timestamps (`2023-01-01 00:00` + 5 min × t, timezone unproven) put the pilot weekday profile's minimum at 01:55 and peaks at 07:45 and 15:45. That fits local-time traffic, so slot = t mod 288 is a defensible feature.
- **Day of week:** reconstructed Sundays are among the lowest days, but two holiday Mondays (2 and 16 January) are equally low and Saturdays are near weekday level. 22 days cannot uniquely pin the weekly phase, so the evidence is too weak.

**Parameter sharing:**
- One parameter set is shared by all sensors. The model is sensor-agnostic: a window from any sensor is processed identically.
- This is what makes the same model usable centralised, locally, and under FedAvg without client-specific parameters. Sensor embeddings would be meaningless to average across clients that own disjoint sensors, so they are excluded from the baseline.

**Size and memory:**

| | B0 (F = 2) | B1 (F = 4) |
|---|---:|---:|
| GRU parameters, 3·(64F + 64² + 2·64) | 13,056 | 13,440 |
| Linear 64 → 12 | 780 | 780 |
| **Total** | **13,836** | **14,220** |
| float32 size | 55,344 B | 56,880 B |
| With Adam state (×3) | ≈ 166 KB | ≈ 171 KB |

- **One batch** of 512 × 12 × F is 49 KB (F = 2); activations are under 1 MB.
- **Series arrays per run:** about 1.4 MB per array for 40 sensors × 8,928 steps.
- **A fully materialised split** would also fit easily (train `X` 24 MB at F = 2). Peak process memory is expected to be dominated by PyTorch itself, a few hundred MB.

---

## 2. Dataset interface

### 2.1 Artifacts consumed (already present; VERIFIED 2026-10-27)

| Path (per run: `data/processed/` primary, `data/processed/missing_validation/` validation) | Field | Shape, dtype | Use |
|---|---|---|---|
| `pilot_preprocessed.npz` | `inputs_scaled` | (S, 8928) float32, finite | feature 0 |
| | `input_imputed` | (S, 8928) bool | feature 1 = `1 − imputed` |
| | `targets_scaled` | (S, 8928) float32, NaN = missing | targets |
| | `observed_mask` | (S, 8928) bool | target mask |
| | `array_index`, `station_id` | (S,) int64 | identity (provisional IDs) |
| | `flow_raw` | (S, 8928) float32 | baselines and raw-unit checks |
| | `tod_median` | (S, 288) float64 | baseline fallback |
| `window_index.npz` | `train` / `val` / `test` | (N, 2) int64 `[sensor_pos, t0]` | split isolation |
| `scaler_parameters.json` | `mean`, `std` (length S), `fit_range` [0, 6336], `mode` | JSON | inverse transform |
| `sensor_selection.csv` / `missing_validation_selection.csv` | `proposed_client` (row = `sensor_pos`; order verified equal to the npz) | CSV | client membership |
| `split_config.json` | boundaries and timestamps | JSON | calendar slot (B1), reporting |

- **Primary run:** S = 40. Windows: train 252,520, val 34,120, test 68,680 (6,313 / 853 / 1,717 per sensor).
- **Validation run:** S = 8. Windows: 50,504 / 6,824 / 13,736. It is the only run with real masked targets (1,452 masked cells across splits).

**Every required field already exists.** No preprocessing change is needed.

### 2.2 Proposed module `aiori/dataset.py` [proposed]
- **`load_run(run_dir)`:** a read-only bundle holding the npz arrays (loaded fully: ≈ 7 MB for S = 40), the window indices, scaler arrays and client labels.
- **`WindowDataset(run, split, sensor_positions=None, calendar=False)`:**
  - **Construction:** selects `run.windows[split]` and, if `sensor_positions` is given (client membership), keeps only those rows. The order is deterministic.
  - **Batch access:** `get_batch(idx_array)` calls the existing `aiori.windows.gather_windows` on the selected `(sensor_pos, t0)` rows. It returns `x` (B,12,F), `y` (B,12) with **NaN replaced by 0.0**, `mask` (B,12) bool, `sensor_pos` (B,) and `t0` (B,). It asserts that `x` and `y` are finite and that `mask` agrees with the original NaNs. Preprocessing is **reused, not duplicated**.
- **Lazy access:** windows are built per batch from the small series arrays (no window tensor stored). This scales to the planned ≈ 200-sensor experiment (≈ 7 MB of arrays).
- **Loader:** a `torch.utils.data.DataLoader` over a `BatchSampler` of indices with `collate_fn = get_batch`.
  - `num_workers = 0`: Windows uses process spawning, and the data is in memory anyway.
  - `pin_memory = False` (CPU).
  - **Train:** shuffled with a seeded `torch.Generator`, batch 512.
  - **Val / test:** not shuffled, batch 4096.
- **Split isolation:** a dataset can only read windows from `windows[split]`. Training code never receives `val` or `test` indices except through explicit evaluation calls.
- **Client membership:** `client_positions(run) → {client: sensor_pos array}` from the manifest's `proposed_client`. For Phase 6 (centralised) all 40 positions are used.

### 2.3 Minimal additions needed (none touch preprocessing)
1. `aiori/dataset.py` (above).
2. A calendar-slot helper for B1, computing slot = t mod 288 from the window's `t0`.
3. A scaler loader returning `mean[S]` and `std[S]` arrays. The existing `preprocessing.inverse_transform(z, scaler, sensors=sensor_pos)` already handles mixed-sensor batches.
4. A prediction-record writer (§4.4).
5. `torch` added to `requirements.txt` after the approved installation.

---

## 3. Training protocol

| Item | Specification | Status |
|---|---|---|
| Loss | masked MSE in z-space: `L = Σ M·(ŷ − y)² / Σ M`, computed as `where(M, ŷ − y, 0)`, never `M·(ŷ − y)` on NaN data | [brief] |
| Batches with Σ M = 0 | no backward pass or optimiser step; counted and logged. Cannot occur on the primary pilot; can occur in principle on the validation run | [brief] |
| Optimiser | Adam, learning rate 1e-3, default betas, no weight decay | [brief] |
| Seeds | 0, 1, 2. Each seed fixes Python, NumPy and PyTorch RNGs, model initialisation and the shuffle generator | [brief] |
| Batch size | **512** (≈ 494 updates per epoch on 252,520 train windows) | [proposed] |
| Max epochs | **50** | [proposed] |
| Early stopping | on **validation** masked MAE (raw units, pooled cells), checked every epoch; patience **5**; restore best weights | [brief] validation-only; numbers [proposed] |
| Gradient clipping | global norm 1.0 | [proposed] |
| Learning-rate schedule | none | [proposed] |
| Determinism | `torch.use_deterministic_algorithms(True)`; fixed `torch.set_num_threads(n)` recorded per run; CPU only | [proposed] |
| Test set | not used for tuning, early stopping, or feature choice (B0 vs B1 is chosen on validation). Evaluated once per approved final configuration | [brief] |

- **Compute estimate (to be measured in the Phase 6 smoke test, not yet measured):** about 13.8k parameters, sequence length 12 and batch 512 on a 24-thread CPU. Each epoch is expected to take seconds to tens of seconds, and a full run (≤ 50 epochs) minutes. The GPU is not needed.
- **Pipeline sanity:** before a full run, the model must overfit a tiny fixed batch (loss → ≈ 0) and run one forward/backward pass on real primary-pilot batches.

---

## 4. Evaluation protocol

### 4.1 Unit conversion
`ŷ_raw = ŷ_z · std[s] + mean[s]` and `y_raw` likewise, with `s = sensor_pos` per window (`preprocessing.inverse_transform(..., sensors=sensor_pos)`). Scaler parameters come from training observations only. Predictions are **not** clipped at 0 for the primary metrics [proposed]; the share of negative predictions is reported.

### 4.2 Metrics (masked; observed targets only)
For each cell (window *w*, horizon *h*) with `M[w,h] = True`: `e = ŷ_raw − y_raw`. All metrics report the number of cells used, `n`.

| Metric | Definition |
|---|---|
| **Pooled (micro)** | `MAE = Σ|e| / n`; `RMSE = √(Σe² / n)` over all observed cells of the split. **Primary.** Each (forecast origin, horizon) pair counts once; target times therefore appear in up to 12 overlapping windows, which is standard for multi-horizon evaluation. |
| Per horizon h = 1 … 12 (5 … 60 min) | pooled over cells at horizon h |
| Per sensor | pooled over that sensor's cells |
| Per client | pooled over the client's cells (micro) **and** the mean of its sensors' MAEs (macro) |
| Sensor-macro overall | mean of per-sensor MAE (equal sensor weight). Secondary; with no missing targets it equals micro MAE, but RMSE differs |
| Normalised (secondary) | MAE in z-units, so high-flow sensors don't dominate (R10 in the audit) |

### 4.3 Reference forecasts (all use only past data or training statistics)
1. **Persistence:** `ŷ(t0+h) = last observed raw input value` (latest observed step in t0−12 … t0−1; if none, the training time-of-day median).
2. **Seasonal-naive (daily):** `ŷ(t0+h) = raw(t0+h−288)` (always before t0 because h ≤ 12). If that value is missing or t0+h−288 < 0, use the training time-of-day median.
3. **Historical average:** training time-of-day median for that sensor and slot (`tod_median`).

Weekly seasonal-naive is excluded: the training period holds only 22 days.

### 4.4 Prediction record (one file per run × split) [proposed]
`predictions_<arm>_<seed>_<split>.npz` containing:
- `sensor_pos` (N,) int32 and `array_index` (N,) int64 (provisional identity);
- `client` (N,) int8 (code ↔ name table stored alongside);
- `t0` (N,) int32;
- `y_hat` (N,12) float32 and `y_true` (N,12) float32 (raw units, NaN where missing);
- `mask` (N,12) bool;
- `y_hat_z` (N,12) float32;
- metadata JSON: config, seed, torch version, thread count, best epoch.

### 4.5 Leakage and missing-target controls
- **Scaler and medians:** fitted on training observations only (VERIFIED).
- **Windows:** belong to the split containing all their targets (VERIFIED). Inputs use only past steps.
- **Selection and tuning:** early stopping and B0/B1 choice use validation only. Test metrics are produced once per frozen configuration, and test predictions are never inspected during development.
- **Missing targets:** contribute to neither loss nor metrics; `n` is reported everywhere.
- **Missing inputs:** filled causally and flagged by feature 1.
- **The validation run** (`missing_validation/`) is evaluated separately and **never pooled** with primary metrics.

---

## 5. Federated compatibility (design only; FedAvg is not part of Phase 6)

| Aspect | Specification |
|---|---|
| Arms | Centralised (all 40 sensors' windows); local-only (4 models, each on one client's sensor positions); FedAvg IID-like (random balanced 10/10/10/10 split of the same 40, seeded); FedAvg geographic (`proposed_client`) |
| Shared interface | the same `ForecastGRU`, `WindowDataset(run, split, sensor_positions)`, loss, metrics and prediction record in every arm. Only the set of sensor positions and the update schedule differ |
| Aggregated parameters | all `state_dict` float tensors: `gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.bias_ih_l0`, `gru.bias_hh_l0`, `head.weight`, `head.bias`. No buffers exist |
| Client-local | raw and scaled series, window indices, per-sensor scalers, time-of-day medians, optimiser state (Adam re-initialised at each round start [proposed]), RNG streams |
| Scalers | applied and inverted inside the client. Never sent to the server. Clients report metric *sums* (Σ\|e\|, Σe², n) so the server can compute exact pooled metrics without receiving predictions or scalers |
| Aggregation weights | `n_k` = number of client k's training windows with ≥ 1 observed target [proposed]. In the pilot every client has 63,130, so weights are equal |
| Communication | serialise the `state_dict` (`torch.save` to an in-memory buffer) at every downlink and uplink; log bytes per client, direction and round, plus the cumulative total. Report as **model-payload bytes** (no protocol overhead). Reference: 55,344 B of raw float32 per model (B0) |
| Fairness | same architecture, initial weights per seed, optimiser, batch size and learning rate; equal total local-epoch budget; same validation-based stopping rule; identical test windows and evaluation code; per-client and pooled reporting; 3 seeds, mean ± std |

---

## 6. Acceptance criteria

### 6.1 Checkable now, without a model (status 2026-10-27)

| Criterion | Status |
|---|---|
| Processed artifacts present with the required fields, shapes and dtypes (§2.1) | VERIFIED (inspection; `scripts/verify_artifacts.py` checks, 85/85 PASS read-only) |
| Window targets stay inside their split; inputs start ≥ 0 | VERIFIED |
| Scaler is train-only; inverse round trip is exact | VERIFIED |
| `observed_mask` equals the finite targets; targets are NaN exactly where missing | VERIFIED |
| Manifest order equals npz order (client membership by position) | VERIFIED |
| Time-of-day convention is plausible on training data | SUPPORTED (§1) |
| Synthetic preprocessing tests | VERIFIED: 55 passed (2026-10-27) |

### 6.2 Requires the Phase 6 implementation
1. The loader reads both runs; `x`, `y` and `mask` shapes and dtypes are as specified; `x` and `y` are finite; the mask agrees with the original NaNs.
2. The loader for `split = s` yields only `windows[s]` rows; the client filter yields only the requested positions.
3. The forward pass maps (B,12,F) to (B,12) for F ∈ {2, 4}; the parameter count is 13,836 / 14,220.
4. The masked loss equals a NumPy reference; missing targets carry no gradient; it is finite when targets are NaN.
5. A fully masked batch returns "skip" with no NaN and no optimiser step.
6. Inverse scaling is correct for mixed-sensor batches against a NumPy reference.
7. The same seed gives identical weights after N steps and identical metrics on CPU.
8. A CPU forward/backward smoke test runs on real primary and validation-run batches.
9. Tiny-batch overfit: loss drops below a small threshold.
10. Leakage guards: no code path passes `val`/`test` windows to the optimiser; the test split is evaluated only when explicitly requested.
11. The full test suite passes (existing 55 + new).
12. **Separate approval is obtained before full pilot training.**
