# AIORI 2.0 Phase 9A: Historical Traffic Replay Map

Implementation report, 2026-11-15.

## A. Status

**PASS.**

All 15 acceptance criteria were met on this workstation. The limits are listed in section I:
- sensor locations are provisional;
- there is no congestion classification;
- the feature runs on a separate local server, port 8010, so production is unchanged.

---

## B. Existing project audit

**Front end.**
- The front end is React 19, TypeScript 7, Vite 8 and Tailwind 4.
- It uses React Router 8 and Recharts 3, in `src/app/web`.
- It had no map library. I added Leaflet 1.9.4 (MIT) with OpenStreetMap tiles. It is small, framework-free, and its canvas renderer suits 200 markers that change every frame.

**Back end.**
- The back end is a Python standard-library HTTP server, `src/app/backend/server.py`.
- It runs PyTorch on CPU.
- It loads files once at start-up, with no database.
- Production runs it on port 8000, behind the keep-alive task and the Cloudflare quick tunnel. Both serve `web/dist`.

**Data.**
- Source: `src/data/processed/aiori2_3yr`, from Phase 2, using the conservative missing-value policy.
- `flow_raw.npy` holds 200 sensors × 315,648 steps.
- The coverage runs from 2022-01-01 00:00 to 2024-12-31 23:55, in 1,096 days.
- Timestamps come from `metadata/calendar/timestamp.npy`. At start-up it is checked to be a regular 5-minute grid.
- Time zone:
  - naive Pacific wall-clock time, inferred by the Phase 2 audit;
  - nothing is shifted;
  - on 2024-03-10, slots 02:00–02:55 (the clock hour skipped by daylight saving) have no readings and are flagged.

**Geographic metadata.**
- Each sensor's `station_id` from `metadata/sensors.csv` is joined to `Lat`/`Lng` in `sensor_meta_feature.csv`:

  | Check | Result |
  |---|---|
  | Unique join | one-to-one |
  | Sensors with coordinates | 200 of 200 |
  | All inside the California bounding box | yes |
  | Districts agree with the study's district | 200 of 200 |
  | Sensor pairs sharing one location | 2 pairs |

- The study's row-to-station mapping (`node_order[array_index]`) is "supported but unconfirmed". Every location is therefore labelled **provisional** in the API and the UI.

**Checkpoint compatibility.**
- Each Stage 3 FL `best.pt` holds `models["global"]`, the aggregated FedAvgM model.
- Input contract: `(B, 12, 1)` scaled flow → `(B, 12)` scaled flow.
- Scaling uses the run's client scaler scope:
  - `client_geo` for geographic runs;
  - `client_iid` for IID runs.
- Raw output is `z · std[sensor] + mean[sensor]`.
- Inputs are `experiments/conservative/A/filled_inputs.npy`:
  - causal forward fill of at most 6 steps;
  - then the 2023 training time-of-day median.
- This is exactly the contract used for training and evaluation, so no adaptation was needed.

---

## C. Implemented components

**Back end (new files, `src/app/backend/`).**
- `replay.py` is the `Replay` service.
- `replay_server.py` is the entry point:
  - it reuses `server.Engine` and `server.Handler` unchanged, so every original route still works;
  - it adds `GET /api/replay/*`;
  - it serves `web/dist-replay`;
  - it listens on port 8010.

| Endpoint | Returns |
|---|---|
| `/api/replay/meta` | coverage, per-day coverage for 1,096 dates, periods, model provenance, data, time, location and congestion notes |
| `/api/replay/sensors` | 200 sensors: anonymised ID (the same IDs as the original app), region, IID client, type, district, county, freeway, provisional lat/lng. No station IDs. |
| `/api/replay/day?date=` | per-slot count of sensors with a valid reading (288), DST slots, period |
| `/api/replay/frame?time=&layer=observed` | 200 values or `null`, status `ok`/`missing`/`invalid`, relative flow |
| `/api/replay/frame?time=&layer=forecast&partition=&horizon=` | batched 200-sensor forecast for T + horizon, actuals, `no_input` status |
| `/api/replay/series?sensor=&date=` | one sensor's 288 readings for a day, plus its 2023 typical curve |
| `/api/replay/forecast?sensor=&time=&partition=` | inputs (observed, model input, imputed flags), 12 forecasts, 12 actuals with status, per-step absolute error, MAE and WAPE for that hour, warnings, provenance |

Request handling and safety:
- Inputs are validated with strict regular expressions:
  - 5-minute boundary;
  - within the dataset;
  - known sensor ID;
  - `geo`/`iid` only;
  - horizon 1–12;
  - unknown or repeated parameters are rejected.
- Checkpoints are fixed at start-up and are never selectable by a request.
- Errors return JSON with status 400, 404 or 422. Internals return 500 without details.
- Caching is bounded LRU: 512 frames and 2,048 forecasts.
- Arrays are memory-mapped. A request reads only one column, one row slice or a 12-step window.

**Front end.**
- New files:
  - `src/pages/Replay.tsx`;
  - `src/components/replay/ReplayMap.tsx` and `replay.css`;
  - `src/lib/replay.ts` and `src/lib/replay.test.ts`.
- What the page provides:
  - a date picker, with previous and next day;
  - a time slider with a coverage strip (bars for sensor counts, red ticks for slots with no readings);
  - Play and Pause, previous and next reading, and a speed control of 1, 2, 4 or 8 steps per second;
  - a period and DST label;
  - a map with an Observed / Predicted layer and a lead time from +5 to +60 minutes;
  - a sensor panel (map click or list) showing the day's readings against the typical curve;
  - Forecast next 60 min, with a Geographic (non-IID) / IID switch;
  - a forecast chart (observed input, filled input, forecast, observed afterwards);
  - a step-by-step comparison table, provenance, and a data-notes card.
- How requests and playback behave:
  - Stale requests: each key change aborts the previous request, and a result is shown only if its key still matches.
  - Slider moves are debounced by 60 ms.
  - Playback schedules the next step only after the current frame has arrived, so requests never overlap.
  - Steps with no readings are skipped, with a notice saying how many.
  - Playback stops at the dataset ends and on leaving the page.
- Integration is gated by Vite's built-in `MODE === "replay"`:
  - `npm run build:replay` writes `dist-replay` with the route and the "Historical Traffic Replay" navigation entry;
  - plain `npm run build` produces the original navigation with no `/replay` route. I verified this on a scratch build. The Replay code chunk is still emitted there, but nothing references it.

**Comparison logic.**
- Absolute error is computed only where a valid actual exists.
- MAE and WAPE are computed over those steps. WAPE only when the observed total is positive.
- Steps after 2024-12-31 23:55 are labelled `beyond_dataset`.

**Map colour.**
- The colour shows the reading divided by that sensor's 2023 training median for the same time of day, on a log scale from 0.5× to 2×.
- It is labelled as relative flow, **not a congestion level**.
- A zero reading takes the low end of the scale. Missing or invalid readings are hollow grey markers.

---

## D. Model provenance

**Configuration.**
- The frozen Stage 3 winner, `final_manifest.json`. Its SHA-256 `b3efd3be441a…` is checked at start-up against `final_manifest.sha256`.
- MLP (medium preset), FedAvgM with server momentum 0.9.
- History A: trained on 2023, conservative policy, epoch 20.

| Partition | Model ID | Run | Checkpoint SHA-256 | Scaler scope |
|---|---|---|---|---|
| Geographic (default) | `aiori2-mlp-fedavgm-A-geo-s20261027` | `fedavg_geo-A-mlp-medium-fedavgm-s20261027-ba24d6a0a6` | `cf602d8e10dfa09720887ec39f567332d59b6f958fcb2cc4d6092af5b4c2ab0b` | client_geo |
| IID | `aiori2-mlp-fedavgm-A-iid-s20261027` | `fedavg_iid-A-mlp-medium-fedavgm-s20261027-1387d83baa` | `5691435950af745aa891ac5de45e8e2a9a0fa5387512cc4d1bda3004f22267fe` | client_iid |

**Seed policy.**
- The served seed is 20261027, the first pre-registered seed. It was fixed before any Stage 3 or test result existed, so it was chosen neither by validation nor by test performance.
- The UI shows the three-seed test mean (IID 10.506%, geographic 10.546%) with a note that it is not a guarantee for this seed, sensor or time.

**Checks at start-up.** The service refuses to start if any of these fail:
- the manifest hash;
- the `best.pt` and `run.json` hashes against the manifest;
- the `scalers.json` hash and scope;
- the config's arm, history, model, policy and seed;
- the checkpoint's run ID and its single `global` model.

**Preprocessing artifacts.**
- `experiments/conservative/A/scalers.json`, fitted on 2023-01-01 to 2024-01-01 training data only;
- `filled_inputs.npy` and `input_imputed.npy`;
- `tod_median_train.npy`, used only for the map colour and the typical curve.

**Input/output contract.**
- A forecast "issued at T" uses the readings from T−55 min to T and predicts T+5 to T+60 min. This is the study's window with t0 = T + 5 min.
- Predictions are unclipped, as they were evaluated. A warning appears when any value is negative.
- If all 12 inputs are imputed, the response is 422 "unavailable".
- Before 2023-01-01 00:55 the response is 422, because history A has no inputs earlier than that.

---

## E. Test evidence

| Suite | Result |
|---|---|
| `src/app/tests/test_replay.py`, new | 30 passed |
| `src/app/tests`, all (original backend and candidate tests, plus replay) | 56 passed |
| `src/tests`, research suite (regression) | 502 passed, 2 skipped |
| `npm test` (Vitest) | 22 passed, 8 of them new |
| `npm run typecheck` | clean |
| `npm run build:replay` | built (exit 0). One earlier attempt failed, most likely a file lock while the server was reading `dist-replay`; the rerun passed. |
| Default production build | built to a scratch folder; `dist` was not rebuilt |

The replay tests cover:
- **Time and dates:**
  - dataset boundaries;
  - invalid dates and times (400 or 404);
  - per-day counts equal to the mask;
  - the 12-slot DST gap.
- **Data:**
  - missing readings returned as `null`, never 0;
  - raw-unit series;
  - sensor ID and coordinate alignment against the metadata file;
  - IDs identical to the original app.
- **Model loading:**
  - checkpoint and scaler provenance;
  - eval mode with no gradients;
  - a tampered checkpoint and a tampered manifest are both rejected.
- **Inference:**
  - **equality with the frozen evaluation path:** `aiori2.run.Runner.predict` plus `ThreeYearWindows.inverse` on 256 validation windows per partition, within 0.001 vehicles (float32 rounding from batch-of-256 vs batch-of-1);
  - forecast and actual timestamp alignment;
  - **no future leakage:** a spy confirms the inputs never read past T, and perturbing all data after T leaves the forecast unchanged;
  - missing and beyond-dataset actuals;
  - fully imputed input gives 422;
  - no forecast before 2023;
  - the batched frame equals single forecasts.
- **HTTP:**
  - all endpoints;
  - parameter rejection, including path-like sensor IDs;
  - no paths or station IDs in responses;
  - original `/api/health`, `/api/sensors` and `/api/predict` unchanged.

**Browser checks.** I drove the built page with scripts in the in-app browser. They are not saved as automated end-to-end tests. Results:
- the page loads with Leaflet tiles and 200 markers;
- a real marker click selects the sensor and shows its tooltip;
- the sensor list works;
- the forecast and comparison table match the API;
- Play and Pause work: at 8× speed, 22 frames in 3 s with **0 overlapping requests**, and the time stays fixed after Pause;
- next and previous step, and previous day, work;
- an invalid date shows a message and leaves the time unchanged;
- the DST gap is skipped with a notice;
- 13 rapid slider moves sent 1 request and the final time was displayed;
- the Predicted layer and lead time update the map;
- 2022 shows "forecasts unavailable";
- the IID switch works;
- an all-imputed input shows the unavailable message;
- leaving the page during playback stops all frame requests;
- the original Overview, Forecasting, Performance, Federated, Methodology and About pages render without errors;
- at 375 px wide there is no horizontal overflow.

**Bugs found and fixed during testing.**
- **Poisoned cache from an aborted request.** If a request was aborted while its body was being read, the replay client turned the abort into an empty `{}` "success". That `{}` was cached, and revisiting that time crashed the map. Fixed: the abort now propagates, and nothing is cached after an abort. A regression test was added.
- **Zero readings coloured as missing.** A genuine zero reading was drawn grey like a missing value. It now takes the low end of the scale.

---

## F. Performance

Measured on the workstation, CPU, over HTTP on localhost.

| Operation | Time |
|---|---|
| Replay data and both model loads | 0.1 s (memory-mapped) |
| Single forecast | 2.7 ms median (model forward pass 1.3 ms) |
| Observed map frame, 200 sensors, cold | 16 ms median, 22 ms max |
| Forecast map frame, 200 sensors | 5.6 ms median, 22.5 ms max |
| One sensor's day series | 2.1 ms |
| `meta` | 44 ms (50 kB) |
| `sensors` | 13 ms (57 kB) |

- **Playback:** about 7 frames per second at the 8× setting. Speed is limited by the wait-for-frame rule, not by the server.
- **Memory:** about 840 MB resident for the whole server process. This includes the original Engine, the optional candidate model and memory-mapped arrays.
- **No GPU** is used, and the training watchdog is not touched.

---

## G. Preservation

**Modified (narrow edits, with verified backups in `src/app/backups/phase9a_20261115-102507/`):**
- `src/app/web/src/App.tsx`: +4 lines, a lazy `Replay` route rendered only when `MODE === "replay"`.
- `src/app/web/src/components/layout/Shell.tsx`: +3 −1 lines, one navigation entry, replay mode only.
- `src/app/web/package.json`: added `leaflet`, `@types/leaflet` and the `build:replay` script.
- `src/app/web/package-lock.json`: the matching lock file entries.
- `src/app/web/.gitignore`: added `dist-replay/`.

**New files:**
- `src/app/backend/replay.py` and `replay_server.py`;
- `src/app/tests/test_replay.py`;
- `src/app/web/src/pages/Replay.tsx`;
- `src/app/web/src/components/replay/ReplayMap.tsx` and `replay.css`;
- `src/app/web/src/lib/replay.ts` and `replay.test.ts`;
- this report.

**Build output:** `src/app/web/dist-replay/`, git-ignored.

**Hash status:**
- 38 of 41 protected production files are unchanged. The three that differ (`App.tsx`, `Shell.tsx`, `package.json`) are the intended edits listed above.
- Unchanged and not restarted:
  - `server.py` and `candidate.py`;
  - the original front end;
  - `vite.config.ts` and `index.html`;
  - deploy files;
  - the local preview-server launch configuration (not part of the repository);
  - `.keepalive/keepalive.ps1`.
- Production `web/dist` was not rebuilt; its newest file is from before this phase.

**Training artifacts:**
- All 107 Stage 3 artifacts are byte-identical, checked against a snapshot taken before the work began.
- The frozen manifest hash still matches.
- The watchdog's `auto_resume` is still false.
- Nothing was retrained or tuned, and test results were not used for anything.

**Git:** HEAD is still `dad975f`. Nothing was committed or pushed.

**Rollback:**
- stop `replay_server.py`;
- optionally delete `dist-replay`;
- to remove the code, restore the five backed-up files and delete the new files.

---

## H. Access

From the repository root:

```powershell
cd src\app\web; npm run build:replay; cd ..\..\..        # only after front-end changes
.venv\Scripts\python.exe src\app\backend\replay_server.py
```

- Open **http://127.0.0.1:8010/replay**. Optional: `AIORI_REPLAY_PORT`, and `AIORI_CANDIDATE=1` for the original candidate model.
- The production command (`server.py`, port 8000) is unchanged and does not include the replay route.

How to use the page:
1. Pick a date, or use the arrows. Drag the time slider or press Play. Change the speed in the menu.
2. Click a marker, or choose a sensor from the list.
3. Choose Geographic (non-IID) or IID, then press **Forecast next 60 min**.
4. Switch the map to **Predicted** to see every sensor's forecast at a chosen lead time.

---

## I. Known limitations

- **Locations are provisional.** They depend on the unconfirmed row-to-station mapping. Two pairs of sensors share a location.
- **No congestion levels.**
  - The forecasting data is flow-only.
  - The raw arrays have occupancy and speed channels, but their definitions are unconfirmed and the project has no validated thresholds.
  - The colour is relative flow only.
- **Coverage gaps.**
  - The conservative policy masks missing readings, and all-zero readings in 2022 and 2024.
  - Quality flags such as long constant or zero runs are not masks, so long zero runs can appear as valid zeros. For example, IC-10 reads 0 on a weekday morning in September 2024.
- **Forecast availability depends on the period.**
  - None for 2022.
  - 2023 forecasts are in-sample (the training period).
  - Jan–Jun 2024 is the validation period used for selection.
  - Jul–Dec 2024 replays the test period, which was evaluated once.
  - The UI labels each period.
- **One seed per partition is served.** The single-hour errors shown are not comparable with the aggregate study WAPE.
- **Predictions are unclipped.** Negative values are possible and carry a warning.
- **Map tiles need internet access** (OpenStreetMap). Markers still render without tiles.
- **The global header still says the old study period.** The shared header chip ("Historical data · Jan–Sep 2023") and the sidebar footer, both from the existing Shell, describe the original study, so the replay page relies on its own "Historical replay · not live" label and notes. I did not change them, to keep the Shell edit minimal.
- **Latent issue in the original client.** `lib/api.ts`, a protected file, has the same `res.json().catch(() => ({}))` pattern. It does not cache, so it cannot cause the crash fixed above, but it could show an empty result after an abort. Not changed.
- **No saved end-to-end suite.** Front-end interaction checks were scripted in the browser for this review, not saved as an automated suite.
