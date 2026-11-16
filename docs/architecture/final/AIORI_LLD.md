# AIORI Low-Level Design

Part of the [final architecture](AIORI_FINAL_ARCHITECTURE.md). Paths are relative to `src/`. Line numbers refer to the working tree on 2026-11-15.

## LLD-A · Frontend component structure

[LLD-01 component tree](diagrams/lld-01-frontend-tree/aiori-lld-01-frontend-tree.html)

| Part | Location | Responsibility |
|---|---|---|
| Entry | `app/web/src/main.tsx` | `createRoot(...).render(<StrictMode><App/></StrictMode>)` |
| Router | `app/web/src/App.tsx` | `BrowserRouter`, `StudyProvider`, `Shell`, `Suspense`, lazy routes. `const REPLAY = import.meta.env.MODE === "replay"`: the Replay and Disruptions imports are `null` otherwise |
| Layout | `components/layout/Shell.tsx` | `ROUTES` (nav; replay entries only in replay mode); `Sidebar`; header; `Footer`; focus moved to the page `h1` after navigation; Escape closes the mobile menu |
| Study data | `lib/data.tsx` | `StudyProvider` loads health, metrics, sensors, demo and models in parallel once. `useStudy`; `useRequest` (abortable call wrapper) |
| Study gate | `components/study.tsx` | `StudyGate` renders a page only when study data is ready; `ResultsTable`, `FedAvgDiagram`, `ForecastingPerformance` |
| Study API | `lib/api.ts` | `request()` → `fetch(BASE + "/api" + path)`, JSON, `ApiError`. Methods: `health`, `metrics`, `sensors`, `demo`, `models`, `predict` |
| Replay API | `lib/replay.ts` | `replayApi` (meta, sensors, day, frame, series, forecast). Pure helpers: `slotTime`, `nextAvailable`, `relColor`, `Lru`, `forecastRows`, `parseReplayLink` |
| Explorer API | `lib/disruptions.ts` | `disruptionsApi` (meta, list, event, series, compare, robustness). `listParams`, `seriesRows`, `compareRows`, `replayHref` |
| Request hook | `lib/useRemote.ts` | Keyed fetch. A cache hit returns at once; a new key aborts the previous request; data is set only when the key still matches and the request was not aborted |
| Charts | `components/charts/index.tsx`, `components/replay/ForecastView.tsx` | Recharts: `ForecastChart`, `HorizonChart`, `BarCompare`, `ForecastView` |
| Map | `components/replay/ReplayMap.tsx` | Leaflet map created once per sensor list (canvas renderer, OSM tiles). Marker styles update per frame; click selects a sensor; `ResizeObserver` |
| UI kit | `components/ui/index.tsx` | Button, Card, Field, Select, Segmented, TableWrap, Skeleton, ErrorState, EmptyState, Note |

**State**
- No global store. Each page keeps local React state.
  - Replay: date, slot, layer, partition, horizon, playing, speed, selected sensor.
  - Explorer: filters in state; the selected event in the `?event=` URL parameter.
- Module-level bounded LRU caches survive page revisits within a session.

**Request lifecycle**
1. A user action changes a key.
2. `useRemote` aborts the previous request.
3. `fetch` runs.
4. JSON is parsed. An abort during parsing propagates, and an unreadable 200 response is an error.
5. The result is cached only if the request was not aborted.
6. The UI renders data, a skeleton or an `ErrorState` with Retry.

## LLD-B · Backend request processing

[LLD-02 module diagram](diagrams/lld-02-backend-modules/aiori-lld-02-backend-modules.html)

`user action → page → lib client → fetch /api/... → Handler.do_GET/do_POST → service method → NumPy/PyTorch → dict → json.dumps → 200 JSON → client → React state → render`

**`server.Handler` (`app/backend/server.py:345-432`)**
- **GET routes:**
  - `/health`, `/models`, `/sensors`, `/metrics`, `/demo`, also under `/api/`.
  - Anything else goes to the static handler: files from `FRONTEND`, with SPA fallback to `index.html` for extensionless paths outside `/api`.
- **POST:** `/predict` or `/api/predict`.
  - Body must be 1–20,000 bytes of JSON.
  - 408 on a body timeout.
- **OPTIONS:** CORS preflight only for origins listed in `AIORI_CORS_ORIGINS`.
- **Errors:** `ApiError(status)` → `{error:{status,message}}`. Any other exception → 500 "Internal error"; no stack traces or paths are returned.
- **Headers:** `Cache-Control: no-store` (JSON) / `no-cache` (static), `X-Content-Type-Options: nosniff`.

**`replay_server.Handler` (`app/backend/replay_server.py:109`)**
- Subclass of `server.Handler`.
- `/api/replay/disruptions[...]` → `disruption_route()`.
- `/api/replay/<name>` with name in `ALLOWED` → `route()`.
- Any other `/api/replay/*` path → 404.
- Everything else is delegated to `super().do_GET()`.
- `_query` rejects unknown and repeated parameters (`parse_qs`, `max_num_fields`).
- `ReplayError` → status + message; other exceptions → 500.

**Services**
- `server.Engine`: `predict`, `health`, `models_view`, `metrics`, `demo`.
- `replay.Replay`: `meta`, `day`, `frame`, `series`, `forecast`.
- `disruptions.Disruptions`: `meta`, `list`, `detail`, `series`, `compare`, `robustness`.

Endpoint contracts: [AIORI_API_INVENTORY.md](AIORI_API_INVENTORY.md).

## LLD-C · Forecasting inference

[LLD-03 sequence](diagrams/lld-03-forecast-sequence/aiori-lld-03-forecast-sequence.html) · [LLD-08 checkpoint validation](diagrams/lld-08-checkpoint-validation/aiori-lld-08-checkpoint-validation.html)

| Step | 2023 GRU path (`server.Engine.predict`) | Three-year MLP path (`replay.Replay.forecast`) |
|---|---|---|
| 1. Sensor | id `LA-01`…`IC-50` → array position (`by_id`) | same id scheme → position |
| 2. Time | `parse_time`: 5-minute grid from 2023-01-01; t0 = forecast start | `parse_time`: 5-minute grid from 2022-01-01; issue time T, window t0 = T + 1 |
| 3. Input window | `WindowDataset` row for (sensor, t0) in the validation or test split → x `(1, 12, 4)` | `filled_inputs.npy[pos, e-12:e]` → x `(1, 12)`, plus `input_imputed` flags |
| 4. Missing inputs | preprocessing: forward fill ≤ 6 steps, then training time-of-day median; the observed flag is a feature. Custom readings use the same rule (`custom_features`) | same rule (precomputed); all 12 imputed → 422; partly imputed → warning |
| 5. Scaling | per-sensor train-only mean/std; features: z, observed flag, time-of-day sin/cos | client scaler scope (`client_geo` / `client_iid`): z = (x − mean[s]) / std[s] → `(1, 12, 1)` |
| 6. Model | `ForecastGRU(4, hidden)`: GRU + linear head → 12; final model = local-only seed 1, one checkpoint per client; comparison = centralised seed 0 | `build_model("mlp", "medium", 1)`, `load_state_dict(strict)`, eval, no grad; geo and IID global models |
| 7. Forward pass | under a lock, CPU | under a lock, CPU (`_predict`, also used for 200-sensor frame batches) |
| 8. Inverse scaling | `evaluation.to_raw`; final model clipped at 0 | raw = z·std + mean; not clipped (warning if negative) |
| 9. Timestamps | `forecast_timestamps` t0 … t0+11 | t0 … t0+11; steps after 2024-12-31 23:55 → null, `beyond_dataset` |
| 10. Actuals | `flow_raw[pos, t0:t0+12]` (historical mode only) | `flow_raw` where the valid mask is true; otherwise null with status `missing`/`invalid` |
| 11. Errors | none in the response (the frontend compares) | `abs_error` per step, MAE and WAPE over valid steps only (WAPE needs a positive total) |
| 12. Visualisation | `ForecastChart` (observed hour, forecast, observed afterwards) | `ForecastView` (inputs, filled inputs, forecast, actual, step table, provenance) |

Units: vehicles per 5-minute interval.

## LLD-D · Historical replay

[LLD-04 sequence](diagrams/lld-04-replay-sequence/aiori-lld-04-replay-sequence.html)

- **Date and time:**
  - The date input accepts only dates with coverage.
  - The slot slider runs 0–287, with a coverage strip of valid sensors per slot from `GET /day`.
- **Playback:**
  - `playing` and `speed` (1, 2, 4 or 8 steps per second).
  - The next step is scheduled only after the current frame arrives (`frameReady`), so requests never overlap.
  - `step()` skips slots with no valid reading and moves across days with `adjacentDate`.
  - At the dataset ends it stops with a notice.
- **Cancellation:** `useRemote` aborts the previous frame request on every key change; there is a 60 ms debounce when paused.
- **Frames:** `GET /frame?time&layer[&partition&horizon]`.
  - Observed layer: 200 values, status and relative flow (value ÷ 2023 time-of-day median, when the median is ≥ 1).
  - Forecast layer: one batched forward pass for all sensors; sensors whose input is fully imputed get status `no_input`.
- **Server cache:** 512 frames and 2,048 forecasts (LRU).
- **Boundaries:**
  - Forecasts are available from 2023-01-01T00:55.
  - The DST spring-forward slots have no valid readings.
  - Missing or invalid values are `null`, never 0.
- **Deep links:**
  - `parseReplayLink` validates date, time (5-minute boundary, matching date), sensor id and event id.
  - Invalid parts are ignored and reported.
  - A valid event id shows "Back to candidate event".

## LLD-E · Historical disruption detection

[LLD-05 pipeline](diagrams/lld-05-disruption-pipeline/aiori-lld-05-disruption-pipeline.html) · [LLD-06 investigation](diagrams/lld-06-disruption-sequence/aiori-lld-06-disruption-sequence.html)

| Stage | Function (`app/backend/disruption_detect.py`) | Behaviour |
|---|---|---|
| Load | `_inputs` (420) | memory-maps flow, valid mask, timestamps, experiment splits, sensors; training window = 2023 |
| Day types | `holiday_dates` (89), `day_types` (100) | weekday-like / weekend-like; each US federal holiday and the day after Thanksgiving is classed by its own 2023 traffic level |
| Baseline | `build_baseline` (131) | per sensor × day type × slot: median and 1.4826·MAD of valid 2023 readings, pooled with neighbouring slots; spread ≥ √median; support ≥ 30 |
| Score | `score` (164), `flag` (211) | z and relative deviation for valid readings with typical flow ≥ 5; anomalous if z ≤ −4.93 or ≥ +5.33 **and** ≥ 25% from typical |
| Calibration | in `detect` (311) | thresholds = median over sensors of each sensor's 0.1% / 99.9% z quantile over clean 2023 readings |
| Quality | `slot_quality` (195) | bits: zero_reading, stuck_run (≥ 1 h same value), near_missing (≤ 1 h from a gap), level_shift (> 5% of a sensor-month anomalous) |
| Temporal grouping | `segments` (221) | same direction; gaps ≤ 2 readings; ≥ 3 anomalous readings |
| Multi-sensor grouping | `group` (235) | same dataset region and direction, overlapping within 15 min; quality-dominated segments form per-sensor episodes (24 h join) |
| Chance check | `chance_coordination` (251) | random whole-day shifts per sensor: 718 observed vs ~256 expected multi-sensor groups |
| Ids | `event_id` (287) | `YYYYMMDDTHHMM-REGION-D/U-hash6`; hash of config, region, direction, interval and sensors |
| Categories | `categorize` (292), `confidence` (302) | sensor_quality, coordinated_deviation, sustained_deviation, sudden_decrease/increase; confidence high/medium/low |
| Write | `build` (455) | `baseline.npz`, `catalog.json`, `catalog.sha256` written atomically; `events_sha256` stored |
| Load at runtime | `disruptions.Disruptions.__init__` (80) | verifies file hash, version, config hash, events hash, input-data hashes, baseline hash; builds index arrays; raises `Unavailable` otherwise |
| APIs | `list` (149), `detail` (218), `series` (256), `compare` (291), `robustness` (347) | bounded, validated; LRU of 256 |
| Robustness | `_forecast_block` (315), `_errors` (325), `_reference_days` (357) | ≤ 72 causal issue times; targets scored only inside the anomaly and when valid; reference = same times on ≤ 4 nearby same-day-type days with no catalog event |

**Limitations**
- Detection is retrospective (it uses whole days).
- Events are statistical departures, not confirmed incidents.
- Regions are dataset groups, not road links.
- The baseline has no seasonal term.

## LLD-F · Federated training and evaluation

[LLD-07 sequence](diagrams/lld-07-federated-sequence/aiori-lld-07-federated-sequence.html)

| Element | Implementation |
|---|---|
| Partitions | `aiori2/arms.py`: `client_geo` (4 regions) and `client_iid` (random, seed 20261027), 50 sensors each; client scalers fitted on the client's own training data |
| Client construction | `FederatedRun.__init__` (`aiori2/fl/engine.py:126`): one training loader per client (training split only), one global model, one `Algorithm` |
| Local training | `local_train` (`engine.py:83`): fresh SGD optimiser each round, `local_steps` batches from the client's own stream; FedProx adds a proximal term; SCAFFOLD uses control variates |
| Update exchange | in-memory: Δi = wi − w plus extras (step count, SCAFFOLD Δci); payload bytes are logged |
| Aggregation | `aiori2/fl/algorithms.py`: `REGISTRY` of 8 rules; float64 server arithmetic; weights ∝ training windows; server state persists |
| Termination | `Runner` (`aiori2/run.py:218`): 20 epochs × 2 M windows budget; early stopping (patience 3, min delta 1e-4) on validation WAPE |
| Checkpoints | `<run_id>/best.pt`, `latest.pt`, `run.json`, `config.json`, `metrics.jsonl`, written atomically |
| Selection | `aiori2/plan.py`: Stage 1 (40 runs), Stage 2 (56), Stage 3 (24 = 4 finalists × 2 partitions × 3 seeds); ranking by validation only |
| Frozen evaluation | `aiori2/final_eval.py`: `freeze` hashes every file into a manifest; `evaluate` refuses unless hashes match and refuses to overwrite a result |

## LLD-G · Persistence and artifacts

[LLD-09 dependency diagram](diagrams/lld-09-data-artifacts/aiori-lld-09-data-artifacts.html)

| Artifact | Location | Written by | Read at runtime by | Role |
|---|---|---|---|---|
| Three-year arrays, masks, calendar, filled inputs, scalers | `data/processed/aiori2_3yr` | `aiori2.pipeline` | `Replay` (memory-mapped), `disruption_detect` | read-only data |
| Nine-month 2023 data | `data/processed/nine_month` | `aiori` preprocessing | `Engine`, `Candidate` | read-only data |
| Stage 1–3 runs, `final_manifest.json` + `.sha256`, test results | `results/aiori2/phase7_stage1`, `phase8c_stage3` | `aiori2.run`, `final_eval` | `Replay` (hash-checked) | frozen |
| Phase 9 GRU runs, analysis, frozen.json | `results/phase9` | 2023 study | `Engine` | frozen |
| Candidate v1 | `results/accuracy_optimization/candidate_v1` | accuracy study | `Candidate` (opt-in) | frozen |
| Disruption catalog v1 | `results/aiori2/phase9b_disruptions/disruption-detector-v1` | `disruption_detect.py build` | `Disruptions` | generated, versioned |
| Robustness summary | same folder | `disruptions.py robustness` | `Disruptions.meta` | generated, optional |
| Sensor metadata | `sensor_meta_feature.csv` | dataset | `Replay` (provisional coordinates) | read-only |
| Frontend builds | `app/web/dist`, `dist-replay`; `app/web-phase10/dist-phase10` (preview) | Vite | servers (static files) | generated |
| Backups | `app/backups/phase9a_*`, `phase9b_*`, `phase10_*` | manual | — | rollback copies with hash manifests |
| Logs | `.keepalive/*.log`, server stderr | servers, keep-alive | — | operational |

## LLD-H · Operational architecture

[LLD-10 start-up](diagrams/lld-10-runtime-startup/aiori-lld-10-runtime-startup.html)

- **Start-up of `server.py`:** `make_server()` → `Engine()`:
  - torch threads (`AIORI_THREADS`, default 2) and device (`AIORI_DEVICE`, default CPU);
  - nine-month run, sensors, the two GRU models with checkpoint hashes, the optional candidate, frozen JSON.
  - Then `ThreadingHTTPServer` on `AIORI_HOST:AIORI_PORT` (default 127.0.0.1:8000) and `serve_forever`.
- **Start-up of `replay_server.py`:**
  - Sets `AIORI_FRONTEND_DIR` to `web/dist-replay` if it exists (`setdefault`, so an explicit value wins).
  - Then `Engine()` → `Replay()` (timeline and shape checks, memory maps, both models) → `Disruptions()` (or an unavailable reason).
  - Port `AIORI_REPLAY_PORT` (default 8010).
- **Caches:** in-process only. Replay keeps 512 frames and 2,048 forecasts; Disruptions keeps 256 results. All start empty.
- **Logging:** one stderr line per request (method, path, status); no request bodies are logged.
- **Errors:** model or data check failures stop start-up; catalog failures disable only the explorer (503).
- **Keep-alive (`.keepalive/keepalive.ps1`):**
  - Checks `/api/health` every 30 s; after 3 failures it restarts `server.py`. Its kill pattern `*server.py*` also matches `replay_server.py`.
  - Restarts cloudflared and records the new public URL.
  - **Not registered as a task on 2026-11-15.**
- **Shutdown:** stopping a process loses only caches; nothing is written at runtime.
