# AIORI End-to-End Data Flows

Flows 1–6 are runtime web requests. Flows 7–8 are offline batch processes that never run when someone uses the website.

## Flow 1 · Forecast request (Forecasting page)

- **Initiating action:** the user picks a sensor, date and time and presses "Run forecast". The page can instead take 12 readings the user types in.
- **Participants:** `ForecastWorkspace` → `lib/api.ts` → `server.Handler.do_POST` → `Engine.predict` → `WindowDataset` / nine-month arrays → `ForecastGRU` (and `Candidate` if requested).
- **Data exchanged:** request `{sensor, time, models[, observations]}`; response with 12 past values, 12 forecast timestamps, one 12-value forecast per model, and actuals (historical mode).
- **Transformations:**
  1. The sensor id maps to an array position.
  2. The time maps to t0.
  3. The window x is `(1, 12, 4)`: scaled flow, observed flag, time-of-day sin/cos.
  4. The GRU forward pass runs under a lock.
  5. The output is inverse-scaled; the final model is clipped at 0.
- **Error or unavailable states:**
  - 400: bad body, models or readings.
  - 404: unknown sensor, or no historical window. Only the validation and test periods of 2023 have windows.
  - 408: body timeout.
  - In the UI these show as an `ErrorState` with Retry.
- **Output:** `ForecastChart` (observed hour, forecast, observed afterwards), a step table, and an accuracy note.
- **Diagram:** [LLD-03](diagrams/lld-03-forecast-sequence/aiori-lld-03-forecast-sequence.html)

## Flow 2 · Historical Traffic Replay

- **Initiating action:** the user opens `/replay`, picks a date, moves the slider or presses Play.
- **Participants:** `Replay` page → `useRemote` → `replay_server.route` → `Replay.meta/day/frame/series` → memory-mapped arrays.
- **Data exchanged:**
  - meta: coverage per date and model provenance.
  - day: valid sensors per slot.
  - frame: 200 values, statuses and relative flow.
  - series: one sensor's day plus its 2023 typical flow.
- **Transformations:**
  - Values whose validity mask is false become `null`, with status `missing` or `invalid`.
  - Relative flow = value ÷ 2023 time-of-day median.
- **Error or unavailable states:**
  - 404 for dates outside 2022-01-01 … 2024-12-31.
  - Slots with no valid reading are skipped during playback.
  - The DST hour is shown as skipped.
- **Output:** the coloured sensor map, the coverage strip and the day chart.
- **Diagram:** [LLD-04](diagrams/lld-04-replay-sequence/aiori-lld-04-replay-sequence.html)

## Flow 3 · Replay model inference

- **Initiating action:** the user chooses the Predicted layer, or presses "Forecast next 60 min" for one sensor.
- **Participants:** Replay page → `/api/replay/frame?layer=forecast` or `/api/replay/forecast` → `Replay._inputs` → `Replay._predict` → frozen MLP + FedAvgM (geo or IID).
- **Data exchanged:** issue time T and partition. The response carries the 12 input values with imputed flags, 12 forecasts, actuals with statuses, absolute errors, MAE, WAPE, warnings and provenance.
- **Transformations:**
  1. t0 = T + 5 min.
  2. Inputs are `filled_inputs[e-12:e]`, which are causal.
  3. Scaling z = (x − mean)/std uses the client scope.
  4. The MLP forward pass runs.
  5. The result is inverse-scaled.
  6. Future observations are read separately, only for comparison.
- **Error or unavailable states:**
  - 422 before 2023-01-01T00:55 (history A inputs).
  - 422 when all 12 inputs are imputed.
  - Steps past the end of the dataset are reported as `beyond_dataset`.
- **Output:** a forecast chart against the observed values; per-hour MAE and WAPE, labelled as not comparable with the study's aggregate WAPE.
- **Diagram:** [LLD-04](diagrams/lld-04-replay-sequence/aiori-lld-04-replay-sequence.html)

## Flow 4 · Historical Disruption Explorer

- **Initiating action:** the user opens `/disruptions`, sets filters and opens an event.
- **Participants:** Explorer page → `disruptionsApi` → `disruption_route` → `Disruptions.list/detail/series/compare` → the in-memory catalog, plus Replay's arrays for readings.
- **Data exchanged:**
  - Filtered, paginated event briefs.
  - Event detail: sensors, why it was flagged, quality flags, replay target.
  - Series: observed values, typical flow, detection band, z, anomalous flags.
  - Comparison: flow ÷ typical for up to 12 sensors.
- **Transformations:**
  - Filtering and sorting use NumPy masks over 16,445 events, in deterministic order.
  - The series window is the event ± 2 h, capped at 3 days.
- **Error or unavailable states:**
  - 400: invalid filters.
  - 404: unknown id or sensor.
  - 503: catalog missing or stale.
  - An empty result shows an empty state.
- **Output:** the overview, the event table and the event panel with charts and limitations.
- **Diagram:** [LLD-06](diagrams/lld-06-disruption-sequence/aiori-lld-06-disruption-sequence.html)

## Flow 5 · Candidate anomaly → Historical Replay

- **Initiating action:** "Open in Historical Traffic Replay" in an event, or "Open this moment in Replay" from its forecast card.
- **Participants:** Explorer (`replayHref`) → React Router → Replay page (`parseReplayLink`).
- **Data exchanged:** URL `/replay?date=YYYY-MM-DD&time=YYYY-MM-DDTHH:MM&sensor=ID&event=ID`. No server call is made for the navigation itself.
- **Transformations:**
  - Each parameter is validated against the dataset dates, the 5-minute grid and the sensor ids.
  - Valid values set the initial date, slot and selected sensor.
- **Error or unavailable states:** invalid parameters are ignored, and a notice lists them.
- **Output:** Replay opens at the event's peak time with the peak sensor selected, and shows "Back to candidate event".
- **Diagram:** [HLD-03](diagrams/hld-03-frontend-features/aiori-hld-03-frontend-features.html), [LLD-06](diagrams/lld-06-disruption-sequence/aiori-lld-06-disruption-sequence.html)

## Flow 6 · Anomaly-specific retrospective forecast comparison

- **Initiating action:** "Compare forecast errors" in an event's forecast card.
- **Participants:** Explorer → `/api/replay/disruptions/<id>/robustness` → `Disruptions._robustness` → `Replay._predict` (batched).
- **Data exchanged:** sensor and partition in; out come the issue range, event-window metrics (n, MAE, WAPE, MAE by horizon, missing targets), reference-day metrics and dates, warnings, method and caution.
- **Transformations:**
  1. Up to 72 issue times run from one hour before the sensor's anomaly to its end.
  2. Each forecast uses only the readings up to its issue time.
  3. Only valid targets inside the anomaly are scored.
  4. The same issue times are repeated on up to 4 nearby same-day-type days with no catalog event at that sensor.
- **Error or unavailable states:**
  - 2022 events return `available: false`.
  - Few targets or no reference day produce warnings.
  - With no valid targets the page shows "Comparison unavailable".
- **Output:** a table and an MAE-by-lead-time chart, labelled exploratory and not a test score.
- **Batch variant:** `disruptions.py robustness` writes `robustness_summary.json` (7,335 events, peak sensor, geo partition), shown as a catalog-wide table.

## Flow 7 · Offline federated model training

- **Initiating action:** an operator runs `python -m aiori2.plan run --stage N` or `python -m aiori2.run --config ...`. This happened on the GPU environment; the website never does it.
- **Participants:** `aiori2.plan` → `aiori2.run.Runner` → `aiori2.arms` (client streams) → `aiori2.fl.FederatedRun` (federated arms) → `aiori2.fl.algorithms` → `Evaluator` → run folder.
- **Data exchanged:** training windows (each client sees its own sensors only), model deltas Δi, global weights, validation metrics, checkpoints.
- **Transformations:**
  - Each round: the global model goes to each client, which runs 100 local SGD steps.
  - The deltas are aggregated by the algorithm's server rule.
  - Epoch-end validation uses validation windows only.
  - `best.pt` is kept by validation WAPE, with early stopping.
- **Error or unavailable states:**
  - Invalid configurations are rejected.
  - Runs are resumable from `latest.pt` and are deterministic.
  - The Stage 3 watchdog cannot relaunch training while `auto_resume` is false.
- **Output:** 96 Stage 1–2 runs (`results/aiori2/phase7_stage1`) and 24 Stage 3 runs (`results/aiori2/phase8c_stage3`).
- **Diagram:** [HLD-04](diagrams/hld-04-ml-federated/aiori-hld-04-ml-federated.html), [LLD-07](diagrams/lld-07-federated-sequence/aiori-lld-07-federated-sequence.html)

## Flow 8 · Frozen model selection and evaluation

- **Initiating action:** an operator runs `aiori2.final_eval freeze`, then `evaluate`, once.
- **Participants:** Stage 3 validation results → `selection.json` → `final_eval` → `final_manifest.json` + `.sha256` → test results → `replay.Replay._load_model` at serving time.
- **Data exchanged:** checkpoint, config and scaler hashes; test metrics per model, client and horizon.
- **Transformations:**
  - Configurations are ranked by validation score; the winner is MLP + FedAvgM, history A.
  - The selected checkpoints are frozen and scored once on Jul–Dec 2024 with the same masks and raw-flow metrics as validation.
- **Error or unavailable states:**
  - `evaluate` refuses if any hash differs or a result already exists.
  - Serving refuses to start on any mismatch (LLD-08).
- **Output:**
  - Test WAPE (3-seed mean): IID 10.506%, geographic 10.546%; persistence 12.495%.
  - Seed 20261027 is served for each partition. It is the first pre-registered seed, not chosen by test results.
- **Diagram:** [HLD-04](diagrams/hld-04-ml-federated/aiori-hld-04-ml-federated.html), [LLD-08](diagrams/lld-08-checkpoint-validation/aiori-lld-08-checkpoint-validation.html)

## Offline: disruption catalog build (supports Flows 4–6)

`python src/app/backend/disruption_detect.py build` produces the versioned catalog in about 14 s: 2023 baseline → thresholds → scoring → segments → regional groups → catalog. See [LLD-05](diagrams/lld-05-disruption-pipeline/aiori-lld-05-disruption-pipeline.html).
