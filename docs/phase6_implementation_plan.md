# Phase 6 implementation plan — compact GRU baseline

**Date:** 2026-10-27. **Status: PROPOSED; not started.** Implements `docs/phase6_technical_spec.md`.

**Scope:** model, masked loss, metrics, reference forecasts, dataset interface, and a centralised training *harness* validated by smoke tests. **Full pilot training is out of scope until separately approved.** FedAvg is a later phase.

**Constraints:**
- No changes to preprocessing, splits, selection, raw or processed data.
- No test-set inspection during development.
- No commit or push without approval.

---

## 1. Dependencies
| Dependency | Purpose | Status |
|---|---|---|
| PyTorch, **CPU** wheel for Python 3.12 / Windows (from the official CPU index) | model, autograd, DataLoader | **needs approval to install**; pin the exact installed version in `requirements.txt` afterwards |
| numpy 2.5.4, pandas 3.1.0, pytest 9.1.1 | existing | installed |
| matplotlib | plots (later phases) | not needed in Phase 6 |

The CPU build is chosen for determinism and simplicity. The Blackwell GPU would need a recent CUDA build, which brings no benefit for a 13.8k-parameter model.

## 2. Proposed source files

| File | Contents | Needs torch? |
|---|---|---|
| `aiori/metrics.py` | `masked_errors`, `pooled_mae_rmse`, `by_horizon`, `by_sensor`, `by_client` (micro + macro), `normalised_mae`; all NumPy on raw-unit arrays plus a mask | no |
| `aiori/baselines.py` | `persistence`, `seasonal_naive_daily`, `historical_average` (spec §4.3) on window indices; NumPy; past data / training statistics only | no |
| `aiori/dataset.py` | `load_run`, `client_positions`, `calendar_features`, `WindowDataset.get_batch` (wraps `windows.gather_windows`; NaN→0 with mask), `make_loader` (BatchSampler, seeded generator, `num_workers=0`) | yes |
| `aiori/model.py` | `ForecastGRU(input_size, hidden=64, horizon=12)`, `count_parameters` | yes |
| `aiori/losses.py` | `masked_mse(y_hat, y, mask) → (loss, n_valid)` using `torch.where`; returns `None` for zero-valid batches | yes |
| `aiori/training.py` | `seed_everything`, `train_one_epoch`, `evaluate(model, loader) → prediction arrays`, `fit` with early stopping on validation MAE, skipped-batch counter, best-weight restore | yes |
| `aiori/evaluation.py` | inverse scaling per window, metric tables, `write_prediction_record` (spec §4.4) | partly |
| `experiments/configs/phase6_b0.json` (+ `phase6_b1.json`) | architecture, features, optimiser, batch, epochs, patience, seeds, thread count | — |
| `experiments/run_phase6.py` | CLI: `--config`, `--seed`, `--smoke` (few batches), `--evaluate val` (default). `--evaluate test` is refused unless `--final` is passed together with an approved config hash | yes |

Existing modules (`splits`, `preprocessing`, `windows`) are **reused unchanged**.

## 3. Implementation order

| Step | Work | Depends on | Complexity |
|---|---|---|---|
| 0 | Approve open decisions (§6) and the PyTorch CPU install | — | S |
| 1 | `metrics.py` + tests (no torch, can start before install) | — | S |
| 2 | `baselines.py` + tests; compute baselines on **validation only** as reference numbers | 1 | S |
| 3 | Install PyTorch CPU; pin the version; confirm the 55 existing tests still pass | 0 | S |
| 4 | `dataset.py` + unit and real-artifact tests | 3 | M |
| 5 | `model.py`, `losses.py` + tests | 3 | S |
| 6 | `training.py`, `evaluation.py` + smoke, overfit and determinism tests | 4, 5 | M |
| 7 | `run_phase6.py --smoke` on the primary run and on the missing-data validation run (CPU, a few batches) | 6 | S |
| 8 | **Stop: report and request approval for full pilot training** (3 seeds; B0, and B1 if approved) | 7 | — |

Estimated effort for steps 1–7: **about 2–4 working days.** S = under half a day, M = 1–2 days.

## 4. Tests

### 4.1 Unit tests (synthetic)
| Test | Checks |
|---|---|
| `test_metrics.py` | pooled MAE/RMSE match a hand calculation; masked cells ignored; per-horizon, per-sensor and per-client micro/macro; `n` reported; all-masked input handled |
| `test_baselines.py` | persistence uses the last *observed* input; seasonal-naive reads only t0+h−288 < t0; fallback to the time-of-day median; no access beyond t0−1 (perturbing values ≥ t0 leaves the forecasts unchanged) |
| `test_dataset.py` | shapes (B,12,F)/(B,12)/(B,12); feature 1 = 1 − imputed; NaN→0 only where the mask is False; split isolation (rows ⊆ `windows[split]`); client filter; calendar slot = t mod 288 per input step; deterministic batch order per seed |
| `test_model.py` | output (B,12) for F = 2, 4; parameter counts 13,836 / 14,220; no buffers in `state_dict` |
| `test_losses.py` | equals a NumPy reference; NaN targets give a finite loss; zero gradient from masked cells; zero-valid batch returns `None` |
| `test_training.py` | zero-valid batch → no optimiser step and counter incremented; early stopping restores the best epoch; tiny-batch overfit (loss → below threshold); same seed and threads → identical weights and metrics; inverse scaling on a mixed-sensor batch matches NumPy |
| `test_leakage_guards.py` | the training loop receives only train windows; the runner refuses `--evaluate test` without `--final` |

### 4.2 Integration tests (real artifacts; marked `data`; skipped automatically when `data/processed/` is absent, e.g. on a fresh clone)
- Load the primary and validation runs. Batch shapes, dtypes and finiteness; mask equality with `targets_scaled` NaNs; window counts 252,520 / 34,120 / 68,680 and 50,504 / 6,824 / 13,736.
- A forward/backward smoke test on one real batch per run. The validation run's batches must include masked targets.
- A reference-forecast smoke test on the validation split only.
- `scripts/verify_artifacts.py` checks still pass (read-only invocation).

## 5. Risks and mitigations
| Risk | Mitigation |
|---|---|
| NaN leaking into the loss (`NaN × 0 = NaN`) | `torch.where`-based loss; NaN→0 in the loader plus a mask; dedicated tests |
| Non-determinism (threads, library versions) | fixed seeds and thread count; deterministic algorithms; record the torch version and threads in every run |
| PyTorch install issues on Windows / Python 3.12 | CPU wheel from the official index; verify the import before any code is written |
| Accidental test-set use | runner guard (`--final`); test predictions not written in development runs; review checklist |
| `sensor_pos` vs `array_index` confusion in inverse scaling | single helper; mixed-sensor test; prediction record stores both |
| Validation over-use (B0/B1, patience) | limit choices to those pre-declared in configs; log every validation-based decision |
| Negative flow predictions | report their share; no clipping in primary metrics (spec) |
| Errors dominated by high-flow sensors | report the normalised and per-sensor metrics too |
| Pilot results over-interpreted | label results as pipeline-pilot; geographic conclusions deferred to the ≈ 200-sensor experiment (research decisions §4) |

## 6. Open decisions (need approval)
| # | Decision | Recommendation |
|---|---|---|
| D-A | Install PyTorch (CPU build) | approve |
| D-B | Feature set: B0 only, or B0 + B1 (time-of-day sin/cos) compared on validation | run B0; include B1 as a pre-declared validation-selected variant |
| D-C | Batch 512, max 50 epochs, patience 5, grad-clip 1.0, no LR schedule | approve |
| D-D | Early-stopping metric = validation pooled MAE in raw units | approve |
| D-E | No clipping of negative predictions in primary metrics | approve |
| D-F | FedAvg design defaults for later: `n_k` = training windows; Adam re-initialised each round | record now, confirm in the FedAvg phase |
| D-G | Commit the new docs and `scripts/analysis/` (note: `original_results/r1_results.json` contains sensor-level IDs) | your call |
