# AIORI Component Inventory

Paths are relative to `src/`. "Runtime" means the component runs while serving web requests. "Offline" means it is a batch job run by an operator.

## Frontend (runtime, in the browser)

| Component | Purpose | Path | Inputs → outputs | Depends on | Endpoints | Diagram |
|---|---|---|---|---|---|---|
| App / router | routes, lazy pages, replay-mode gating | `app/web/src/App.tsx` | URL → page | React Router, StudyProvider, Shell | — | LLD-01 |
| Shell | navigation, header, footer, focus handling | `app/web/src/components/layout/Shell.tsx` | route → layout | `useStudy` | health (status) | LLD-01 |
| StudyProvider / StudyGate | load study data once; gate pages | `lib/data.tsx`, `components/study.tsx` | — → health, metrics, sensors, demo, models | `lib/api.ts` | 5 study GETs | HLD-03 |
| Overview page | headline metrics, example forecast, findings | `pages/Overview.tsx` | study data → cards, chart | ForecastChart | predict | LLD-01 |
| Forecasting page | historical or custom forecast | `pages/Forecast.tsx` | sensor, time, readings → chart, table | `api.predict` | predict | LLD-03 |
| Performance / Federated / Methodology / About | frozen study evidence and method | `pages/*.tsx` | metrics → charts, tables | charts, study components | via StudyProvider | HLD-03 |
| Replay page | timeline, playback, map, sensor panel, forecast | `pages/Replay.tsx`, `components/replay/*` | date/slot/sensor → frames, series, forecasts | `replayApi`, `useRemote`, Leaflet | /api/replay/* | LLD-04 |
| Disruption Explorer | catalog, filters, event detail, charts, replay link, robustness | `pages/Disruptions.tsx` | filters/event → list, detail, series | `disruptionsApi`, `replayApi`, `useRemote` | /api/replay/disruptions*, forecast | LLD-06 |
| API clients | typed fetch, errors, abort | `lib/api.ts`, `lib/replay.ts`, `lib/disruptions.ts` | params → typed JSON | fetch | all | HLD-02 |
| useRemote | keyed fetch, abort, LRU | `lib/useRemote.ts` | key → data/error/loading | `Lru` | — | LLD-04 |

## Backend (runtime)

| Component | Purpose | Path | Inputs → outputs | Depends on | Endpoints | Artifacts | Diagram |
|---|---|---|---|---|---|---|---|
| server.Handler | HTTP routing, static files, errors, CORS | `app/backend/server.py:345` | HTTP → JSON/static | Engine | study API | frontend dir | LLD-02 |
| server.Engine | GRU inference, study views | `app/backend/server.py:85` | request dict → response dict | `aiori` dataset/evaluation/model, torch | study API | nine_month, results/phase9 | LLD-03 |
| Candidate | opt-in optimised MLP forecasts | `app/backend/candidate.py` | (pos, t0) → 12 values | `aopt` package | predict (`candidate`) | results/accuracy_optimization/candidate_v1 | LLD-02 |
| replay_server.Handler | replay/disruption routing and validation | `app/backend/replay_server.py:109` | HTTP → JSON | server.Handler, Replay, Disruptions | /api/replay/* | dist-replay | LLD-02 |
| replay.Replay | observations, frames, forecasts, provenance | `app/backend/replay.py:94` | time/sensor → JSON | `aiori2.models.factory`, `aiori2.arms`, torch, pandas | replay endpoints | aiori2_3yr, phase8c_stage3, sensor_meta_feature.csv | LLD-04, LLD-08 |
| disruptions.Disruptions | catalog queries, series, compare, robustness | `app/backend/disruptions.py:79` | filters/id → JSON | Replay, disruption_detect | disruption endpoints | catalog v1, baseline.npz | LLD-06, LLD-08 |

## Offline components

| Component | Purpose | Path | Inputs → outputs | Diagram |
|---|---|---|---|---|
| aiori2.pipeline | three-year preprocessing | `aiori2/pipeline.py`, `aiori2/core.py` | XTraffic files → aiori2_3yr | HLD-04, LLD-09 |
| aiori2.data / arms | window loading, client streams, metrics | `aiori2/data.py`, `aiori2/arms.py` | arrays → batches | HLD-04 |
| aiori2.models | MLP, GRU, TCN, PatchTST, N-HiTS | `aiori2/models/` | (B,12,C) → (B,12) | HLD-04 |
| aiori2.fl | federated engine + 8 algorithms | `aiori2/fl/engine.py`, `algorithms.py` | client streams → global model | LLD-07 |
| aiori2.run | experiment runner, evaluator, checkpoints | `aiori2/run.py` | config → run folder | LLD-07 |
| aiori2.plan / track / baseline | stage plans, tables, persistence baseline | `aiori2/plan.py`, `track.py`, `baseline.py` | run results → selection | HLD-04 |
| aiori2.final_eval | freeze manifest, one-time test | `aiori2/final_eval.py` | selection → manifest, results | HLD-04 |
| Stage 3 watchdog | guards the Stage 3 batch (no launch while `auto_resume` is false) | `aiori2_watchdog/` | status files → status | HLD-05 |
| aiori package | 2023 GRU study (preprocessing, training, FedAvg, evaluation) | `aiori/` | nine-month data → phase9 runs | HLD-04 |
| Accuracy optimisation | optimised MLP candidate | `experiments/accuracy_optimization/` | nine-month data → candidate_v1 | LLD-09 |
| disruption_detect | anomaly detector, catalog build/verify CLI | `app/backend/disruption_detect.py` | arrays → catalog | LLD-05 |
| disruptions robustness CLI | catalog-wide exploratory forecast comparison | `app/backend/disruptions.py:491` | catalog + models → robustness_summary.json | LLD-05 |

## Operations and deployment files

| Component | Path | State on 2026-11-15 |
|---|---|---|
| Keep-alive watchdog | `.keepalive/keepalive.ps1`, `install-task.ps1` | script present; task not registered |
| Cloudflare quick tunnel | started by keep-alive | not running |
| Stage 3 watchdog task | `\AIORI2\AIORI2_Stage3_Watchdog` | Ready; `auto_resume` false |
| Docker image recipe | `app/backend/Dockerfile` | file present; not verified as in use |
| Hugging Face deploy script | `deploy/hf_deploy.py`, `deploy/hf-space/` | file present; not verified as in use |
| Vercel config (frontend only) | `app/web/vercel.json` | file present; not verified as in use |
| Legacy static frontend | `app/frontend/` | served only if no `web/dist` exists |

> **Superseded (2026-11-16).** The final deployment is `deploy/aiori` (Docker Compose, served publicly through a
> Cloudflare Quick Tunnel). The items above are legacy files kept for history; `.keepalive/` stays on the local
> machine and is not part of the repository.

## Tests

| Suite | Path | Covers |
|---|---|---|
| App backend | `app/tests/test_backend.py`, `test_candidate.py` | study API, predict validation, candidate |
| Replay | `app/tests/test_replay.py` | indexing, provenance, inference equivalence, no leakage, HTTP |
| Disruptions | `app/tests/test_disruptions.py` | detector on synthetic fixtures, catalog integrity, causal forecasts, HTTP |
| Research | `tests/test_*.py` | preprocessing, leakage guards, models, federated engine and algorithms, phase analyses, watchdog |
| Frontend | `app/web/src/lib/*.test.ts` (Vitest) | clients, time helpers, colours, caches, chart rows, deep links |
