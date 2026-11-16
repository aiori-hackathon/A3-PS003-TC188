# AIORI API Inventory

Verified against `src/app/backend/server.py` (Handler 345–432, Engine 85–339), `replay_server.py` (route 91, disruption_route 61, Handler 109), `replay.py` and `disruptions.py`. All responses are JSON unless stated.

**Errors**
- All errors have the shape `{"error": {"status": <int>, "message": <str>}}`.
- Unexpected failures return 500 "Internal error" without internal details.
- No endpoint returns station ids, array rows or file paths. Tests check this for meta and sensors.

## Study API (port 8000 and port 8010)

Every path also works without the `/api` prefix.

| Endpoint | Method | Parameters | Response | Validation / errors | Service | Frontend consumer |
|---|---|---|---|---|---|---|
| `/api/health` | GET | none | status, device, torch version, models loaded, sensor count, uptime, candidate_enabled | — | `Engine.health` | `StudyProvider`, Shell status |
| `/api/models` | GET | none | deployed models with provenance (arm, seed, architecture, selection, validation/test MAE, checkpoint hash prefix) | — | `Engine.models_view` | `StudyProvider` |
| `/api/sensors` | GET | none | 200 sensors: id, client, client name, type group, median test flow | — | `Engine.sensors` | `StudyProvider`, Forecasting sensor picker |
| `/api/metrics` | GET | none | frozen 2023-study test aggregates (arms, baselines, contrasts, runtime, percentage error, optimised-MLP results) | — | `Engine.metrics` | Performance, Federated, Overview |
| `/api/demo` | GET | none | curated historical examples (4 regions × 4 situations) | — | `Engine.demo` | Overview, Forecasting examples |
| `/api/predict` | POST | JSON body: `sensor`; `time` (`YYYY-MM-DDTHH:MM`); `models` (subset of `final`, `centralised`, and `candidate` if enabled); optional `observations` (12 values or null) | sensor, mode (historical/custom), 12 past values, 12 forecast timestamps, forecasts per model, actual (historical), period, accuracy note, inference ms | 400: body empty or > 20 kB, invalid JSON, bad models, bad observations, candidate with custom readings. 404: unknown sensor or no historical window. 408: body timeout | `Engine.predict` | Forecasting, Overview example |
| `/api/predict` | OPTIONS | CORS preflight | 204 for allowed origins, else 403 | origins from `AIORI_CORS_ORIGINS` | Handler | — |
| `/` and other paths | GET | — | static files from the frontend folder; extensionless paths outside `/api` get `index.html` | missing files with an extension → 404 | `Handler._static` | browser |

## Replay API (port 8010 only)

Query parameters only. Unknown or repeated parameters return 400. Unknown `/api/replay/<name>` returns 404.

| Endpoint | Parameters | Response | Validation / errors | Service | Consumer |
|---|---|---|---|---|---|
| `/api/replay/meta` | none | source, units, timezone note, coverage per date, periods, first forecast time, location and congestion notes, model provenance (geo, IID) | — | `Replay.meta` | Replay, Disruptions (via sensors) |
| `/api/replay/sensors` | none | 200 sensors: id, region, IID client, type, district, county, freeway, lat/lng or null, `location_status`, `shares_location` | — | `Replay.sensors` | Replay map, Explorer filters |
| `/api/replay/day` | `date` | valid-sensor count per slot (288), DST slots, period label, available slots | 400 bad format; 404 outside dataset | `Replay.day` | Replay coverage strip |
| `/api/replay/frame` | `time`; `layer` = observed/forecast; `partition` = geo/iid; `horizon` 1–12 | 200 values (null = no reading), status, relative flow; forecast layer adds target time, actual, model id | 400 bad time, layer, partition or horizon; 422 forecast before 2023-01-01T00:55 | `Replay.frame` (LRU 512) | Replay map |
| `/api/replay/series` | `sensor`, `date` | 288 timestamps, values, status, 2023 typical flow | 404 unknown sensor | `Replay.series` | Replay sensor panel |
| `/api/replay/forecast` | `sensor`, `time`, `partition` | inputs (model input, observed, imputed flags), 12-step forecast, actual + status, abs error, MAE, WAPE, warnings, provenance, inference ms | 404 unknown sensor/time; 422 before forecasts start or all inputs imputed | `Replay.forecast` (LRU 2,048) | Replay, Explorer |

## Disruption API (port 8010 only)

Every route returns 503 with the rebuild command if the catalog is missing, stale or fails a hash check.

| Endpoint | Parameters | Response | Validation / errors | Service | Consumer |
|---|---|---|---|---|---|
| `/api/replay/disruptions/meta` | none | version, creation time, config + hash, thresholds and calibration, holiday rules, baseline method, provenance (file names only), summary, audit incl. chance check, categories, quality-flag texts, limitations, robustness summary | 400 if any parameter | `Disruptions.meta` | Explorer overview, limitations |
| `/api/replay/disruptions` | `from`, `to` (dates); `region` (LA/BA/CS/IC, comma list); `category`; `direction`; `confidence`; `sensor`; `min_sensors`; `min_duration`; `max_duration`; `min_peak_z`; `quality` and `holiday` (include/exclude/only); `sort` (±start, ±score, ±peak, ±duration, ±sensors); `page`; `page_size` ≤ 100 (default 25) | total, page, pages, items (brief events) | 400 unknown parameter or bad value; 404 unknown sensor; deterministic ordering | `Disruptions.list` | Explorer list |
| `/api/replay/disruptions/<id>` | none | full event, sensor details, category and quality texts, why flagged, period, replay target, forecast availability | 404 malformed or unknown id | `Disruptions.detail` | Explorer event panel |
| `/api/replay/disruptions/<id>/series` | `sensor` (required) | window (event ± 2 h, ≤ 3 days), observed (null if missing), status, typical flow, detection band, z, anomalous flags, quality flags, DST slots | 400 missing sensor; 404 sensor not in event | `Disruptions.series` (LRU 256) | Explorer flow chart |
| `/api/replay/disruptions/<id>/compare` | none | up to 12 sensors: flow ÷ typical per step, availability | 404 unknown id | `Disruptions.compare` | Explorer comparison |
| `/api/replay/disruptions/<id>/robustness` | `sensor` (required), `partition` | issue times (≤ 72), event-window and reference-day MAE/WAPE/n by horizon, reference dates, warnings, method, caution; or `available: false` for 2022 | 400 bad partition; 404 sensor not in event | `Disruptions.robustness` | Explorer forecast comparison |

**Event id format:** `\d{8}T\d{4}-(LA|BA|CS|IC)-[DU]-[0-9a-f]{6}` (validated before lookup). Example: `20230820T1025-LA-D-26d2a2`.

## Not present

- No write endpoints other than `POST /api/predict`, which stores nothing.
- No authentication, no websockets and no streaming endpoints.
