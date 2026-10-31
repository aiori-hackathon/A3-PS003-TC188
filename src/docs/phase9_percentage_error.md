# Phase 9 — percentage error of the frozen models (read-only analysis)

**Date:** 2026-10-29. **Source:** the frozen Phase 9 held-out test predictions (8 Aug – 30 Sep 2023, 3,098,409 windows, 37,174,578 observed target cells), written once by the Phase 9 test stage.

**How it was computed:** `app/analysis/percentage_error.py`.
- It runs no model, changes no frozen file, and writes only the aggregate `app/data/percentage_error.json`.
- The test was **not** re-run, and nothing was tuned on it.
- The MAEs recomputed from the same predictions equal the Phase 9 report exactly (16.990, 17.134, 17.274, 17.362).

## Definitions
- **WAPE** = 100 × Σ|actual − predicted| / Σ|actual|, over every observed target cell. It is MAE divided by the mean observed flow (195.84 vehicles per 5 min).
- **sMAPE** = 100 × mean of 2|actual − predicted| / (|actual| + |predicted|), excluding cells where both actual and predicted are 0.
- **Seeds:** arm values average the three seeds per cell, as in Phase 9.
- **MAPE is not reported.** 1.02% of observed cells are exactly 0, where MAPE is undefined, and 4.44% are below 5 vehicles, where it explodes.
- **Persistence and the other reference forecasts:** their WAPE follows exactly from their frozen MAE and the same cells. Their sMAPE needs per-cell predictions, which were not saved, so it is not available.

## Results

| Model (test, three-seed mean) | WAPE | sMAPE | MAE |
|---|---:|---:|---:|
| **Local-only GRU, clipped at 0 (pre-declared final model)** | **8.672%** | **17.145%** | 16.984 |
| Local-only GRU | 8.675% | 17.608% | 16.990 |
| Centralised GRU | 8.749% | 17.633% | 17.134 |
| IID FedAvg | 8.820% | 17.688% | 17.274 |
| Geographic FedAvg | 8.865% | 17.890% | 17.362 |
| Persistence | 12.470% | n/a | 24.421 |
| Same time yesterday | 16.498% | n/a | 32.309 |
| Historical average | 18.268% | n/a | 35.776 |

The deployed single checkpoint (local-only seed 1, clipped) has WAPE 8.673%, sMAPE 17.197% and MAE 16.985.

**Final model by client group:**

| Client group | WAPE | sMAPE | Persistence WAPE |
|---|---:|---:|---:|
| LA / Ventura | 7.78% | 16.14% | 11.10% |
| Bay Area | 8.90% | 12.55% | 12.72% |
| Coastal / South Central | 8.10% | 18.10% | 12.27% |
| Inland / Central | 10.57% | 21.82% | 14.73% |

**Final model by horizon:**
- **WAPE:** 6.65% at 5 min, 7.63% at 15 min, 8.61% at 30 min, 9.46% at 45 min, 10.37% at 60 min.
- **sMAPE:** 14.04% at 5 min, rising to 20.03% at 60 min.
- **Persistence WAPE:** 7.66% at 5 min, 17.14% at 60 min.

## The 6% WAPE target
**Not achieved.**
- **Gap:** the final model's WAPE is 8.672%, 2.67 percentage points above the target.
- **Short horizons too:** even the 5-minute-ahead horizon alone is 6.65%. Reaching 6% overall would need lower error at every horizon, not only the far ones.
- **What the other numbers mean:**
  - MAE 16.984 is in vehicles per 5 minutes; it is not a percentage.
  - "About 30% better than persistence" is an improvement, not an error rate.

## Possible future improvements
These are proposals only; none was attempted, and any of them would need new data for a fair test:
1. Longer input history (2–3 hours) and day-of-week or holiday features. Currently the model sees only 60 minutes and the time of day.
2. Spatial context from neighbouring sensors on the same freeway, for example a graph or attention model. Each GRU now sees one sensor at a time.
3. Tuning on validation only: hidden size, learning rate, horizon-weighted or WAPE-aligned loss.
4. Seed ensembling (averaging the three seeds' forecasts), selected on validation.
5. Evaluating any change on genuinely new data (for example October–December 2023). The January–September test split has now been used once and cannot judge further changes.
