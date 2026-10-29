# AIORI Traffic Intelligence (app)

A demo and results dashboard for AIORI PS-003. It makes real next-hour traffic-flow forecasts with the study's frozen Phase 9 GRU checkpoints. It shows only results verified on the held-out test.

- **Back end:** `app/backend/server.py`, a Python standard-library HTTP server running real PyTorch inference.
- **Front end:** `app/web`, built with React 19, TypeScript, Vite, Tailwind CSS 4, React Router, Recharts and Lucide icons.
- **Earlier front end:** the dependency-free version stays in `app/frontend` as a fallback.

## Run locally (Windows)
**Prerequisites:**
- the project's virtual environment (Python 3.12, PyTorch 2.14.1 CPU, NumPy, pandas);
- Node.js 22 LTS or newer, which is needed only to build the front end (verified with Node 26.13.0 LTS and npm 11.20).

Build the front end once:
```powershell
cd app\web
npm ci
npm run build
```

Then start the service from the repository root:
```powershell
.venv\Scripts\python.exe app\backend\server.py
```
Open http://127.0.0.1:8000.
- The server serves `app/web/dist` when that build exists, and falls back to `app/frontend` when it does not.
- To force the earlier front end, set `AIORI_FRONTEND_DIR=app\frontend`.

**Front-end development** (hot reload; the Vite dev server proxies `/api` to the Python server on port 8000). Start the Python server first, then:
```powershell
cd app\web
npm run dev
```
The app opens on http://localhost:5173.

**Tests:**
```powershell
.venv\Scripts\python.exe -m pytest app\tests     # API, inference equivalence, privacy, future-forecast labelling
cd app\web; npm test                              # CSV parsing, clock arithmetic, chart series
npm run typecheck
```

**Local research artifacts:** the server needs these git-ignored files:
- `data/processed/nine_month/`: the Phase 9 checkpoint (scaled inputs, training-only scaler, window index).
- `results/phase9/runs/local_seed1/` and `results/phase9/runs/centralised_seed0/`: frozen weights and run metadata.
- `results/phase9/final/`: `analysis.json`, `frozen.json` and the test metrics.

**Environment variables:**

| Variable | Default | Purpose |
|---|---|---|
| `AIORI_HOST` | `127.0.0.1` | bind address |
| `AIORI_PORT` | `8000` | port |
| `AIORI_DATA_DIR`, `AIORI_RUNS_DIR`, `AIORI_FINAL_DIR` | the paths above | artifact locations |
| `AIORI_FRONTEND_DIR` | `app/web/dist`, else `app/frontend` | folder served at `/` |
| `AIORI_CORS_ORIGINS` | none (cross-origin calls refused) | comma-separated origins allowed to call the API, needed only when the front end is hosted elsewhere |
| `AIORI_DEVICE` | `cpu` | `cuda` is optional; CPU needs about 1–3 ms per forecast |
| `AIORI_THREADS` | `2` | PyTorch CPU threads |
| `AIORI_CANDIDATE` | off | `1` serves the optimized MLP (`results/accuracy_optimization/candidate_v1`) for historical replay; 12-reading forecasts keep the original GRU. The demo launch config sets it. |
| `VITE_API_BASE` | empty | build-time setting for the front end: the API origin when the API runs on a different host |

## What is served
Every route also answers under `/api/...`; the React app uses the `/api` form.

| Endpoint | Purpose |
|---|---|
| `GET /health` | service status, device, models loaded |
| `GET /models` | provenance of the deployed models: seed, selection rule, validation and test MAE of that seed, checkpoint hash prefixes |
| `GET /sensors` | 200 anonymised sensors (for example `LA-01`), client group, provisional type, median test-period flow |
| `GET /metrics` | aggregate held-out results, comparisons, runtime and communication, percentage errors |
| `GET /demo` | fixed historical examples from the held-out period |
| `POST /predict` | `{"sensor": "LA-10", "time": "2023-09-12T07:30", "models": ["final", "centralised"]}` replays history; adding `"observations": [12 values or null]` forecasts from your own readings |

Each forecast response says what kind of forecast it is:
- `mode`: `historical` or `custom`;
- `validated`: true only for windows in the held-out test period;
- `beyond_study_period`: true when any forecast step falls after 30 September 2023;
- `accuracy_note`: a sentence the interface shows next to every forecast.

## Forecasting
The Forecasting page has two modes, kept visibly separate.
- **Historical hour:** replays a window from the validation or held-out test period, 11 July to 30 September 2023. It shows the readings observed afterwards for comparison. Held-out results are the only ones with measured accuracy.
- **Future forecast:** takes the latest 12 five-minute readings for one sensor, oldest first, and forecasts the following 12 intervals for any date from 2023 to 2099. Forecasts after September 2023 are labelled "not validated".
  - The model was trained on January to July 2023, and nothing later was ever observed or evaluated.
  - The model's only calendar input is time of day, so the same readings at the same clock time give the same forecast on any date.
- **Entering readings:**
  - Type the readings in, or upload a CSV with a `timestamp` and a `flow` column; a header row is optional.
  - Times can be written as `2024-03-05 07:55` or `2024-03-05T07:55`. Commas, semicolons or tabs all work as separators.
  - The latest 12 rows are used, and they must be consecutive 5-minute intervals.
  - Leave a value blank (or write `NA`) when it is missing. It is filled the way the model was trained: the last value for up to 30 minutes, then the sensor's training time-of-day median.
  - A blank template can be downloaded from the page.
- **Readings must come from the chosen sensor.** Each sensor is scaled with its own 2023 training statistics.

## Models
- **Original GRU (Phase 9 final model):** the local-only GRU, seed 1, which is the local-only seed with the lowest validation MAE. The local-only arm was the pre-declared final model, chosen on validation before the test.
  - Each sensor uses its own client's model.
  - Negative predictions are clipped at 0, which was the validation-selected policy.
  - This seed's test MAE is 16.991, or 16.985 clipped. The dashboard tables report three-seed means.
- **Comparison model:** the centralised GRU, seed 0 (lowest centralised validation MAE).
- **Historical requests** use the study's own batch builder (`aiori.dataset`), so their inputs are identical to training. The tests check that the API's forecasts equal the frozen held-out predictions.
- **Custom readings** rebuild the same four features: scaled flow with causal fill, the observed flag, and time of day as sine and cosine.

## Percentage error (verified, read-only)
- **Original GRU:** Aug 8 – Sep 30, 2023 held-out test WAPE 8.672% and sMAPE 17.145%; on Oct – Dec 2023, 8.834%.
- **Optimized MLP** (1-day context, trained independently per client, not federated): validation WAPE 6.794%; one-time independent Oct – Dec 2023 WAPE 7.669%, 13.2% lower than the original GRU (8.834%) and 39.5% lower than naive persistence (12.678%) on the same windows. Details: `docs/accuracy_optimization_report.md`.
- **Target:** the 6% WAPE target has not been reached by any evaluated model; the best independent result is 7.669% (optimized MLP, Oct – Dec 2023).
- **Details:** `docs/phase9_percentage_error.md`.

## Privacy and integrity
- Responses never contain station IDs, array rows, file paths or checkpoints.
- Historical data is labelled historical; user readings are labelled as such; nothing is live.
- Frozen research artifacts are only read, never written.
- The front-end source contains no random or stand-in data; a test checks this.

## Design
- **Direction:** a calm, premium developer-platform dark interface, set by the project lead with the Supabase site and dashboard as the quality reference. No layouts, logos or assets were copied.
- **Design system:** tokens, type scale, components and breakpoints are in `DESIGN.md`; product context is in `PRODUCT.md`. The React app keeps these tokens as Tailwind theme variables in `app/web/src/index.css`.
- **Look:**
  - near-black and charcoal surfaces with 1px borders;
  - one muted emerald accent for the highlighted model, the primary action and the active route;
  - amber for real observations shown afterwards, and for the "not validated" caution on future forecasts;
  - Geist (self-hosted, SIL Open Font License) with tabular numerals for data.
- **Components:** shadcn/ui-style primitives in `app/web/src/components/ui` (buttons, cards, tags, fields, switch, segmented control, tables, skeleton, error and empty states). Charts are built with Recharts and follow the same palette.
- **Checked at:** 375, 768 and 1280 px with no horizontal overflow; keyboard navigation (skip link, focus moves to the page heading on navigation, focusable charts, a drawer that is inert when closed); loading, error and empty states. The Impeccable detector reports no findings.

## Deployment
**Not publicly deployed.** A public deployment needs the project lead's approval: the API host must hold the private model checkpoints and processed data, and those must never be published. Two supported layouts:

1. **Single host (simplest).** Build `app/web`, then run the Python server on a small VM or container with the artifacts. It serves the API and the React build on one origin.
   - `app/backend/Dockerfile` builds an image that **mounts** the artifacts read-only at run time instead of copying them in. It has not been built here, because Docker is not installed on this workstation.
   - Put it behind HTTPS (for example a reverse proxy), and set `AIORI_HOST=0.0.0.0` when not using the image.
2. **Static front end plus separate API.**
   - Build with `VITE_API_BASE=https://<api-host>` and publish `app/web/dist` to any static host. SPA rewrites are included for Netlify (`public/_redirects`) and Vercel (`vercel.json`).
   - Run the API as in option 1 with `AIORI_CORS_ORIGINS=https://<front-end-host>`.

The static front end alone contains no data or checkpoints; every number comes from the API at run time.
