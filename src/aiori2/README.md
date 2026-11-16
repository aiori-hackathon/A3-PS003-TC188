# aiori2 — three-year XTraffic preprocessing (AIORI 2.0, Phase 2)

Flow-only, leakage-free preprocessing of XTraffic 2022–2024 for the **same 200 sensors** and **4 × 50 clients** as the
2023 study. It never writes to `data/processed/nine_month/`, `results/` or the application; outputs go to
`src/data/processed/aiori2_3yr/` (git-ignored).

## Run (Windows or Linux, from `src/`)

```
python -m aiori2 all                      # protect, extract, assemble, quality, experiments, manifest, verify
python -m aiori2 verify                   # re-hash outputs and the protected 2023 artifacts
python -m aiori2 --base-dir /data/src --out /data/aiori2_3yr experiments manifest verify
```

Steps are resumable: `extract` skips months whose source (size, mtime) and selection are unchanged and whose
output hash matches. Files are written to `*.partial` and renamed, so an interrupted run leaves no half file.
Settings live in `aiori2/configs/preprocess_v1.json` (paths are relative to `src/`).

## Outputs

| Path | Content |
|---|---|
| `extracted/<year>/<year>_pMM.npy` | (200, T_month, 3) float32 reference copy of flow/occupancy/speed, selection order |
| `extracted/flow_raw.npy` | (200, 315648) float32 raw flow 2022-01-01 00:00 … 2024-12-31 23:55 (NaN and zeros preserved) |
| `metadata/calendar/*.npy` | reconstructed timestamp, year, month, day, weekday (Mon=0), slot, DST-nonexistent hour |
| `metadata/sensors.csv`, `timeline.json` | position, array_index, station_id, geographic and IID client; timeline facts |
| `masks/<policy>/valid_target.npy`, `suspected_missing.npy` | per-policy target masks |
| `flags/*.npy` | non-finite, all-three-zero, ≥24 h zero run, ≥24 h constant run, ≥24 h suspected outage (per policy) |
| `experiments/<policy>/<A|B>/` | `experiment.json`, `scalers.json`, `filled_inputs.npy`, `input_imputed.npy`, `tod_median_train.npy`, `windows/{train,val,test}.npy` |
| `reports/` | `quality_summary.json`, `quality_by_sensor_year.csv` |
| `manifests/` | `extraction.json` (source shape/dtype/SHA-256), `protected_baseline.json`, `dgx_transfer_manifest.json`, `run_log.json` |

## Rules

- **Timestamps are reconstructed** from file names and 288 five-minute slots per day; Pacific local time is inferred,
  not verified. DST slots are neither shifted nor removed; the missing spring-forward hour is left to the masks.
- **Missing data.** `conservative` (default): non-finite flow, and every flow = occupancy = speed = 0 reading in 2022/2024.
  `outage`: non-finite flow, and only all-zero runs of at least 12 steps (1 h) in 2022/2024. Neither is ground truth.
  The same policy must be used for every arm and model in a comparison.
- **Quality flags are not masks**: constant or zero runs stay valid targets unless a policy masks them.
- **Splits**: A trains on 2023, B on 2022–2023; both validate on Jan–Jun 2024 and test on Jul–Dec 2024. A window
  `(sensor, t0)` has inputs `[t0-12, t0)` and targets `[t0, t0+12)`; targets lie entirely inside the split, inputs may
  precede the split but never the experiment's first day. Window index `t0` is relative to the experiment's first
  timestamp (`timeline_offset` in `experiment.json`); validation/test windows are identical for A and B on the timeline.
  A window is kept when at least `min_valid_targets` (1) of its 12 targets are valid — the same rule for every arm.
- **Train-only statistics**: time-of-day medians, fill fallbacks and scalers use valid training observations only
  (`aiori.preprocessing`). Inputs are forward-filled for at most 6 steps (never across a split start), then the
  training time-of-day median. Targets are never filled.
- **Scaler scopes** (`scalers.json`): `per_sensor`, `pooled`, `client_geo`, `client_iid`. Defaults by arm:
  centralised → pooled, local and geographic FedAvg → client_geo, IID FedAvg → client_iid. Pooled scaling uses all
  200 sensors' training data (centralised only); client scopes keep each client's statistics to its own sensors.
  Predictions are inverted with `raw = z * std[sensor] + mean[sensor]`, and MAE/RMSE/WAPE are computed on raw flow
  over the policy's valid targets only (`aiori2.data.raw_metrics`).

## Loading windows

```python
from aiori2.data import ThreeYearWindows, raw_metrics
ds = ThreeYearWindows("data/processed/aiori2_3yr", "conservative", "B", "train", scaler_scope="client_geo")
X, Y, M, sensor = ds.get(range(0, 4096))     # X (B,12,1) scaled flow, Y (B,12) raw flow (NaN = invalid), M mask
```

## DGX transfer

`manifests/dgx_transfer_manifest.json` lists the minimum data (default policy), code files and SHA-256 sums.
Copy those files, keep the relative layout, and run `python -m aiori2 verify` there. Raw yearly arrays are not needed.

## Tests

`python -m pytest src/tests/test_aiori2.py` (synthetic end-to-end tests; real-data checks run when outputs exist).
