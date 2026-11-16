# AIORI 2.0 Phase 9B: Historical Disruption Explorer

**Implementation report.** Workstation run on 2026-11-15. Nothing was committed or pushed.

## A. Final status

**Feature status: PASS.**

All 19 completion criteria are met with evidence below. One operational finding sits outside this feature. The production server on port 8000 is not running, and it was already down before Phase 9B began:
- Its logs end on 2026-11-05.
- The machine rebooted on 2026-11-14.
- No keep-alive scheduled task is registered.

I did not restart it. Restarting would change production, the tunnel and the watchdog, which this phase is not authorized to touch. The production *code and build* are verified intact (section K). Starting the service is your call.

## B. Repository and preservation audit

| Item | Finding |
|---|---|
| Repository | `C:\Users\PC-017\Desktop\Traffic`, branch `docs/update-live-link`, HEAD `dad975f`; all Phase 9A/9B work is uncommitted |
| Architecture | React 19, TypeScript, Vite 8, Tailwind 4, React Router, Recharts, Leaflet front end in `src/app/web`. Standard-library HTTP backend in `src/app/backend`: `server.py` serves production on port 8000; `replay_server.py` serves the isolated extended app on port 8010 |
| Phase 9A status | Verified working before any edits: 56 app tests passed, and Replay was checked in the browser |
| New baseline | `src/app/backups/phase9b_20261115-143346/` holds 15 hash-verified copies, including every Phase 9A file and every integration file touched here. It also records SHA-256 for 108 Stage 3 files and 8 core data and experiment files (`stage3_and_data_sha256.json`) |
| Compatibility constraint | `server.py` and the production `dist` stay untouched. The feature lives only in `replay_server.py` and in `--mode replay` builds (`dist-replay`) |

## C. Detection methodology

Implemented in `src/app/backend/disruption_detect.py`. All parameters live in `CONFIG`, and their SHA-256 is stored in the catalog.

**Inputs**
- Raw flow comes from `extracted/flow_raw.npy`, shape 200 × 315,648.
- The project's conservative validity mask is `masks/conservative/valid_target.npy`.
- Timestamps come from `metadata/calendar/timestamp.npy`: naive Pacific wall-clock time, on a regular 5-minute grid.
- Sensor order and IDs follow the Replay scheme (LA-01 …).
- Missing or invalid readings are never scored, and a missing reading is never treated as zero.
- An exact zero that the mask marks as valid *is* scored, but it carries a quality flag (see below).

**Baseline (training year only)**
- For each sensor × day type × five-minute slot, the baseline is the median of the valid 2023 readings, pooled with the neighbouring slots (15 minutes in total).
- The spread is robust: 1.4826 × MAD, floored at √median (count noise).
- A slot needs at least 30 supporting readings; otherwise it has no baseline.

**Day types**
- Day types are weekday-like and weekend-like.
- Each US federal holiday, plus the day after Thanksgiving, is classed by its own 2023 traffic level. The level is the median over sensors of daily mean flow ÷ 2023 mean. Ordinary weekdays sit at 1.035 and weekends at 0.933.
- Weekend-like: New Year, MLK Day, Presidents' Day, Memorial Day, July 4, Labor Day, Thanksgiving, the day after Thanksgiving, Christmas.
- Weekday-like: Juneteenth, Columbus Day, Veterans Day.
- I added this after the audit showed minor holidays producing large false "coordinated" rises.
- Seasonality and 2022/2024 demand drift are **not** modelled (see J).
- Readings are not scored where the baseline is below 5 vehicles per 5 minutes. This avoids unstable ratios near zero.

**Anomaly score per reading**
- z = (x − median) / spread, and the relative deviation is (x − median) / median.
- A reading is anomalous when **both** of these hold:
  - z ≤ −4.93 or z ≥ +5.33;
  - it is at least 25% from typical.

**Threshold calibration (2023 only)**
- For each sensor, take the 0.1% and 99.9% quantiles of z over its scored 2023 readings that have no quality flag. The threshold is the median of those 200 per-sensor quantiles.
- The per-sensor interquartile ranges are −6.28 to −3.65 for the low threshold and 4.38 to 7.42 for the high one.
- Audit history:
  - A first version pooled all readings. Its thresholds came out at −11.4 / +18.9 because 2–3 drifting sensors supplied about 90% of the pooled tail, which made the detector blind to daytime drops. I replaced it with the per-sensor median.
  - Forecast errors and 2024 data were never used for calibration. A test checks that changing data after the training window leaves the thresholds unchanged.
- The 25% floor, the 15-minute minimum and the 0.1% tail are declared practical-significance choices; they were not tuned.

**Grouping**
- **Segments.**
  - Anomalous readings of one sensor in one direction join across gaps of up to 2 readings (10 minutes).
  - A segment needs at least 3 anomalous readings (15 minutes), so 1–2 reading spikes are dropped.
  - A missing-data gap longer than 10 minutes always splits a segment.
- **Events.**
  - Clean segments of the same direction, in the same *dataset region* (the study's geographic client group, not road connectivity), that overlap or fall within 15 minutes of each other form one event.
  - Quality-dominated segments are never grouped across sensors. Instead, one sensor's quality-dominated segments within 24 hours form a single sensor-quality episode.
- **Chance check.** Shifting each sensor's segments by a random whole number of days (5 repeats) gives about 256 multi-sensor groups of 3 or more sensors, against 718 observed. So roughly a third of "coordinated" groups could arise by chance alone; the page says so.

**Quality flags** (per reading, carried through to sensors and events)

| Flag | Meaning |
|---|---|
| `zero_reading` | An exact-zero reading |
| `stuck_run` | The same non-zero value for an hour or more |
| `near_missing` | Within an hour of missing or invalid data |
| `level_shift` | In that calendar month, the sensor is anomalous for more than 5% of its readings |

- An event where at least 50% of anomalous readings are flagged becomes `sensor_quality`.
- Confidence is high, medium or low, and is kept separate from magnitude.

**Categories**
- They are assigned in this order:
  1. `sensor_quality`;
  2. `coordinated_deviation` (3 or more sensors);
  3. `sustained_deviation` (60 minutes or more);
  4. `sudden_decrease` or `sudden_increase`.
- These are descriptions, not causes.

**Score**
- The score is the sum of min(|z|, 10) × 5/60 over the anomalous readings ("capped z-hours").
- It measures magnitude only. It is not a probability.

**Known false-positive risks**
- The 2023 baseline has no seasonal component. 2022 has more events: 7,105, against 4,377 in 2023 and 4,963 in 2024.
- Night hours (00–06) hold 33% of non-quality events in 25% of the day, because low flows make 25% swings easier.
- Holiday classification is binary and country-wide; for example, Presidents' Day looks weekday-like in the Bay Area. 473 non-quality events fall on holidays, and a holiday filter is provided.
- Some sensors are noisy: LA-19 has 425 non-quality events and IC-21 has 374, against a median of 60.

## D. Catalog summary (real data)

- **Location:** `src/results/aiori2/phase9b_disruptions/disruption-detector-v1/`.
- **`catalog.json`** (14.8 MB): SHA-256 `1dbbf9a1…683f`, also recorded in `catalog.sha256`.
- **`baseline.npz`:** SHA-256 `ad7141e8…f8a683f`.
- **Event-list hash:** `abefebb262cd7b91…`, stored as `events_sha256`.
- **Optional robustness summary:** `robustness_summary.json`.

**Reproducibility.** Rebuilding into a scratch folder gave an identical event list and an identical, byte-for-byte baseline file. Event IDs look like `20230820T1025-LA-D-26d2a2`: start time, region, direction, then a hash of the configuration, sensors and interval.

| Measure | Value |
|---|---|
| Candidate events | **16,445**. Of these, 12,236 have no dominant quality issue and 4,209 are sensor-quality episodes |
| By year | 2022: 7,105; 2023: 4,377; 2024: 4,963 |
| By category | sudden decrease 4,230; sudden increase 4,012; sustained 3,276; coordinated 718; sensor quality 4,209 |
| By direction | decrease 8,199; increase 8,246 |
| By region | LA 5,470; BA 4,355; CS 3,595; IC 3,025 |
| Duration, non-quality events | median 35 min, p90 145 min, p99 about 6 h 20 min |
| Duration, all events | median 50 min, p90 about 26 h (quality episodes run long; the longest is 398 days) |
| Sensors per event | 14,387 single-sensor; 718 with 3 or more sensors |
| Sensors with any event | 199 of 200 |
| Quality flags on events | level_shift 2,768; stuck_run 1,549; zero_reading 935; near_missing 122 |
| Confidence | high 12,053; medium 183; low 4,209 |
| DST | Both checks are covered by tests: the spring-forward hour has no valid reading, so it can never be anomalous, and missing readings in event series are null, never 0 |

**Example event, cause not verified.**
- `20230820T1025-LA-D-26d2a2`: on 20 Aug 2023, 15 LA-group sensors read up to 78% below typical for 11 h 15 min. The CS group shows a matching event, `…-CS-D-f6f8fe`.
- That date coincides with publicly reported Tropical Storm Hilary in Southern California. This project did not verify that link against any incident data, and the detector did not use it.
- The application labels the event only as a candidate coordinated deviation.

## E. Implemented backend

| File | Role |
|---|---|
| `src/app/backend/disruption_detect.py` (new) | Offline pipeline and CLI. `build` writes the versioned catalog and baseline atomically; `verify` checks the catalog hash |
| `src/app/backend/disruptions.py` (new) | The `Disruptions` service: list, detail, series, compare, robustness and the summary. Its CLI `robustness` writes the bounded offline summary |
| `src/app/backend/replay_server.py` (edited) | Adds `/api/replay/disruptions*` routing. A missing or stale catalog makes only these routes answer 503 |

**Start-up checks.** The catalog loads once at start-up, in 0.58 s, and is never rebuilt during a request. Loading verifies, in turn:
- the file hash;
- the schema, detector version and configuration hash;
- the event-list hash;
- the input-data hashes against the live files;
- the baseline hash.

**Endpoints.** All are GET, and unknown or repeated parameters are rejected.
- `/api/replay/disruptions/meta`: version, method, thresholds, holiday rules, provenance (file names only, no paths), summary, audit, chance check, limitations and the robustness summary.
- `/api/replay/disruptions`: list.
  - Filters: `from`, `to`, `region`, `category`, `direction`, `sensor`, `min_sensors`, `min_duration`, `max_duration`, `min_peak_z`, `confidence`, `quality` (include, exclude or only), `holiday`.
  - Sorting: `sort` by start, score, peak, duration or sensors, each ascending or descending. Ordering is deterministic, with ties broken by start time.
  - Paging: `page`, and `page_size` up to 100 (default 25).
- `/api/replay/disruptions/<id>`: detail, including why it was flagged, sensor details, the replay target (date, time, sensor) and forecast availability.
- `/api/replay/disruptions/<id>/series?sensor=`:
  - Observed values are null where missing; status, typical flow, detection band, z, anomalous flag, scored flag, quality flags and DST slots are included.
  - The window covers the event plus 2 h each side, capped at 3 days around the sensor's peak.
- `/api/replay/disruptions/<id>/compare`: up to 12 sensors on a shared axis of flow ÷ typical, with null where unavailable.
- `/api/replay/disruptions/<id>/robustness?sensor=&partition=`: the exploratory comparison described in G.
- Forecasts reuse the existing `/api/replay/forecast`; no inference was duplicated.

**Errors.**
- 400: invalid input.
- 404: unknown or malformed ID, unknown sensor, or a sensor that is not in the event.
- 503: catalog unavailable; the message includes the rebuild command.
- 500: shown without internal details.

## F. Implemented frontend

**New files**
- `pages/Disruptions.tsx`, at the route `/disruptions`.
- `lib/disruptions.ts`, with its tests in `lib/disruptions.test.ts`.
- `lib/useRemote.ts` and `components/replay/ForecastView.tsx`. Both were moved out of `Replay.tsx` unchanged so the two pages can share them.

**What the page contains**
- **Overview:** totals, quality count, sensors involved, durations, and breakdowns by type, year and region. Every number is computed from the catalog; none is hard-coded.
- **Filters:** date range, type, region, sensor, direction, minimum sensors, minimum duration, minimum |z|, confidence, holidays, sort order, and a three-way switch for quality anomalies (hidden by default, with the count shown).
- **Event list:** a paginated table with keyboard-focusable ID buttons.
- **Empty, loading and error states** are all handled.
- **Event detail:**
  - Summary, a "why it was flagged" note, quality warnings and day-type or holiday tags.
  - A flow chart showing observed flow (with gaps where readings are missing), typical flow (labelled as a reference, not a measurement), the detection band, highlighted anomalous readings, red markers for missing or invalid readings, and the shaded anomaly interval.
  - An affected-sensors table and a normalized multi-sensor comparison, with up to 5 lines and an availability percentage for each sensor.
- **Replay link:** "Open in Historical Traffic Replay" opens `/replay?date&time&sensor&event`.
  - `Replay.tsx` validates every parameter (`parseReplayLink`). Invalid ones are ignored with a visible notice.
  - Replay shows a "Back to candidate event" link.
  - Plain `/replay` behaves exactly as before.
- **Forecasts:**
  - Issue the forecast just before onset or before the peak, using the geo or IID partition.
  - "Open this moment in Replay" is offered.
  - 2022 events show an "unavailable" state.
- **Robustness:** an on-demand comparison of errors inside the anomaly and on reference days, plus the catalog-wide summary table.
- **Limitations card:**
  - States what the explorer shows and what it does not claim.
  - Includes the required statement: "Historical traffic anomalies can be investigated here, but their underlying causes are not verified. These results do not establish forecasting accuracy during earthquakes or other disasters."

**Navigation and isolation**
- "Historical Disruption Explorer" appears only in `--mode replay` builds.
- `App.tsx` now creates the lazy imports for Replay and the Explorer only in replay mode.
- I checked that the production build no longer contains either page's code.
- Previously, the production bundle still emitted the Phase 9A Replay chunk even though it was unreachable. This change tightens that 9A isolation too.

## G. Model integration

**Model**
- It is the frozen Phase 9A deployment: MLP + FedAvgM, history A, seed 20261027.
- Partitions: geo `aiori2-mlp-fedavgm-A-geo-s20261027` and IID.
- Checkpoint and manifest SHA-256 are verified at start-up. The final manifest is `b3efd3be…`, unchanged.
- Nothing was retrained or tuned.

**Inference contract**
- It is the same function as Replay: `Replay._predict`, fed with the 12 causal inputs from `filled_inputs.npy` ending at the issue time T.
- A test checks that `_forecast_block` equals `Replay.forecast` (to within 0.011) and uses the same inputs.
- Forecasts are available from 2023-01-01T00:55 onward. 2022 events show "unavailable", which is also tested.

**Errors**
- MAE and WAPE are computed only over targets that are inside the anomaly interval, have a valid reading and come from usable inputs (not all imputed).
- WAPE is reported only when the observed total is positive. A test checks that no out-of-scope or invalid target is ever scored.
- With no valid targets, the page shows "Comparison unavailable".

**Reference days**
- Reference days repeat the same issue times on the 4 nearest days within 4 weeks that meet all of these conditions:
  - same day type;
  - no catalog event at that sensor;
  - at least 80% of readings valid.
- Sample counts and missing-target counts are always shown.

**Catalog-wide summary**
- Scope: geo partition; 7,335 non-quality events from 2023 onward; peak sensor only; at most 24 issues per event; 2 reference days each.
- Pooled WAPE inside anomalies vs. on reference days:

| Period | Inside anomalies | Reference days |
|---|---|---|
| 2023 (training year) | 32.9% | 10.6% |
| Jan–Jun 2024 (validation) | 33.0% | 9.8% |
| Jul–Dec 2024 (test period) | 32.3% | 10.8% |

- By direction: drops reach about 44% WAPE and rises about 22%.
- The page and the JSON both label this as exploratory, selected by unusual flow (so larger errors are expected by construction), not a test score and not causal.

**No leakage and no tuning**
- The detector never reads forecasts.
- Thresholds come from 2023 only.
- The frozen test results and manifest are unchanged.

## H. Test evidence

| Suite | Result |
|---|---|
| `src/app/tests` (backend, candidate, replay, and the new `test_disruptions.py` with 19 tests) | **75 passed** |
| `src/tests` (research) | **502 passed, 2 skipped** (same skips as before) |
| Vitest (5 files, including the new `disruptions.test.ts`) | **30 passed** |
| TypeScript `tsc --noEmit` | clean |
| `npm run build:replay` | OK |
| Production build into the scratch folder | OK; contains no replay or disruption code; real `dist` untouched (`dist/index.html` still dated 2026-10-31) |

**What the new tests cover**
- **Synthetic fixtures** (unit tests only; never shown in the app):
  - injected drop, sustained drop and 3-sensor rise are each found with the exact start time and category;
  - normal days, a 2-reading spike, a missing block and a low-baseline sensor produce no events, with no division warnings;
  - zero readings are scored and flagged as quality, not treated as missing;
  - event IDs are deterministic and change with the configuration;
  - thresholds depend only on the training window;
  - segment gap rules, scoring and quality bits behave as documented;
  - grouping works as documented.
- **Real catalog:**
  - IDs, ordering, timestamps, durations and sensors are consistent;
  - DST and missing readings are never anomalous, and missing values are null, never 0;
  - the baseline equals an independent recomputation of the 2023 median;
  - forecast-contract equality with Replay;
  - only valid, in-interval targets are scored;
  - 2022 events are unavailable for forecasting;
  - stale or missing catalogs are rejected.
- **HTTP:**
  - filters, sort order, pagination and the empty result;
  - detail, series, compare and robustness;
  - 15 bad-request cases return 400 or 404;
  - no filesystem paths leak;
  - a missing catalog returns 503 while the replay endpoints keep working.
- **Front end:** filter serialization, abort handling, unreadable responses, null-not-zero chart rows, normalization rows, and deep-link building and validation.

**Browser checks** (on 127.0.0.1:8010, built-in browser)
- Overview and list load.
- Clicking an event opens the detail and updates `?event=`.
- Charts render with real data.
- Forecast runs: MAE 98.3, 12 of 12 steps compared, 2.7 ms.
- Robustness renders: 462 pairs inside the anomaly against 1,848 on reference days.
- The Replay deep link opens 2023-08-20 at 16:00 with LA-42 selected, and "Back to candidate event" returns to the event.
- Invalid deep links produce a notice and are ignored.
- Plain `/replay`, Play and Next step still work.
- 2022 event: forecast and robustness show "unavailable".
- Rapid sensor changes: the final selection wins (97 events for LA-42).
- The empty-filter state shows correctly.
- At 375 px wide there is no horizontal overflow.
- All 8 routes render, and the nav includes both new entries.
- The console shows no errors from the current build. Earlier entries came from an old Phase 9A bundle that is no longer served.

## I. Performance (measured on this workstation, CPU)

| Item | Measured |
|---|---|
| Offline catalog build | 13.8–14.3 s wall clock; peak working set 492 MB (peak private 910 MB) |
| Offline robustness summary | 9.3 s for 7,335 events |
| Catalog load at server start | 0.58 s, including verification of the input hashes |
| List request (25 rows, filtered) | 2–3 ms; about 12 KB |
| Event detail | 1.6 ms |
| Series | 1.1 ms in-process; 2.9 ms over HTTP |
| Compare | 1.6–1.9 ms |
| Robustness (45 issues plus 4 reference days) | 6.2 ms in-process |
| Single forecast | 2.6 ms over HTTP; 2.7 ms inference |
| Replay server memory | about 866 MB working set, against about 840 MB for Phase 9A alone |

## J. Scientific limitations

- **No causality.** Events are statistical departures in flow. No accident, closure, weather or disaster records were used, and nothing is labelled as such.
- **Retrospective.** Detection uses whole days of data; it is not a real-time detector.
- **Provisional geography.** Regions are the study's data groups. Coordinates and the row-to-station mapping remain provisional, and no road connectivity is inferred.
- **Baseline.**
  - Seasonality, long-term demand change and 2022/2024 level shifts are not modelled; persistent shifts are routed to the `level_shift` quality flag.
  - Holiday handling is binary and decided from 2023.
  - In-sample effect: 2023 events are judged against a baseline estimated partly from those same days, so 2023 counts are slightly conservative.
- **Calibration.** 2023 only, using pre-declared rates; there is no ground truth to measure precision or recall.
- **Chance.** About a third of multi-sensor groups would be expected by chance co-occurrence.
- **Forecast comparisons.** These are exploratory replays. Their errors are not comparable with the frozen three-seed aggregate test WAPE of 10.5%, and they are not a new test score. Events in 2023 fall in the model's training year.

## K. Production preservation

| Check | Result |
|---|---|
| Stage 3 artifacts and core data (116 files vs. the Phase 9B baseline) | **unchanged** |
| Final manifest SHA-256 | `b3efd3be441a…` matches its `.sha256` |
| Protected production manifest (41 files) | 38 match. The 3 that differ are the documented Phase 9A edits (`App.tsx`, `Shell.tsx`, `package.json`). `App.tsx` and `Shell.tsx` gained further mode-gated edits in Phase 9B, and `package.json` was not touched in Phase 9B |
| Files changed vs. the Phase 9B baseline | `App.tsx` (Explorer route, conditional lazy imports), `Shell.tsx` (one nav entry), `Replay.tsx` (deep links; two blocks moved to shared modules), `lib/replay.ts` (`parseReplayLink`, `EVENT_ID_RE` appended), `replay_server.py` (disruption routes) |
| `server.py`, production `dist`, tunnel, keep-alive, watchdog, scheduled tasks | **not modified** |
| Production service, port 8000 | **Not running, and not running before this phase** (see A). Not restarted |
| Deployment / cloud / Archify / Graphify / retraining | none |
| Commits / pushes | none; HEAD is still `dad975f` |

**Rollback**
1. Stop the replay server.
2. Copy the 5 changed files back from `src/app/backups/phase9b_20261115-143346/`.
3. Delete the new files listed in E and F, plus `src/results/aiori2/phase9b_disruptions/`.
4. Run `npm run build:replay`.

Production needs no rollback.

## L. Access and demonstration

**Start the server**
```
.venv\Scripts\python.exe src\app\backend\replay_server.py
```
The log line reports how many events loaded.

**Open the page:** http://127.0.0.1:8010/disruptions

**Rebuild the catalog (optional; it is already built)**
```
.venv\Scripts\python.exe src\app\backend\disruption_detect.py build
.venv\Scripts\python.exe src\app\backend\disruptions.py robustness
```

**Rebuild the front end:** in `src\app\web`, run `npm run build:replay`.

**Demonstration path**
1. Keep "Hide quality anomalies" on and sort by largest score.
2. Open `20230820T1025-LA-D-26d2a2`.
3. Read "Why it was flagged" and look at the LA-42 chart, which shows the band, the typical flow and the anomaly interval.
4. In the sensor table, pick other sensors.
5. Compare the normalized lines.
6. Click **Open in Historical Traffic Replay**. The map opens at 16:00 on 20 Aug 2023 with LA-42 selected; press Play.
7. Click **Back to candidate event**.
8. In "Model forecasts around this event", choose **Just before onset** and click **Forecast next 60 min**.
9. Click **Compare forecast errors**.
10. Scroll to the catalog-wide table and the limitations card.

## M. Remaining limitations

- **Production service on port 8000 is down** (pre-existing; outside this phase's authority). This is the only open item, and it needs your decision.
- **Re-applying Replay deep links.** Replay reads deep-link parameters when the page opens. Changing only the query string while already on `/replay` does not re-apply them. The Explorer always navigates from another route, so it is not affected.
- **Old header label.** The global header still says "Historical data · Jan–Sep 2023", as noted in Phase 9A. It was left unchanged because it belongs to the original production layout.
- **Not measured.** There is no precision or recall against ground truth, because no verified incident labels exist in the project.
