# AIORI: AI-Powered Traffic Intelligence and Federated Forecasting Platform

**Final unified architecture.** Verified against the working tree on 2026-11-15 (Git HEAD `dad975f`; most of the 2022–2024 work is uncommitted, see the verification report). This file is the entry point; each section links to the detailed documents and the 15 Archify diagrams.

| Document | Content |
|---|---|
| [AIORI_HLD.md](AIORI_HLD.md) | High-level design, diagrams HLD-01 to HLD-05 |
| [AIORI_LLD.md](AIORI_LLD.md) | Low-level design, diagrams LLD-01 to LLD-10 |
| [AIORI_COMPONENT_INVENTORY.md](AIORI_COMPONENT_INVENTORY.md) | Every component with path, inputs, outputs, dependencies |
| [AIORI_API_INVENTORY.md](AIORI_API_INVENTORY.md) | Every HTTP endpoint, verified against the handlers |
| [AIORI_DATA_FLOW.md](AIORI_DATA_FLOW.md) | Eight end-to-end flows |
| [AIORI_DEPLOYMENT_ARCHITECTURE.md](AIORI_DEPLOYMENT_ARCHITECTURE.md) | Workstation runtime, ports, processes, operations |
| [AIORI_ARCHITECTURE_VERIFICATION.md](AIORI_ARCHITECTURE_VERIFICATION.md) | Evidence, checks, uncertainty, preservation |

## 1. What AIORI is

AIORI forecasts freeway traffic flow 60 minutes ahead (12 steps of 5 minutes) from the previous hour (12 readings) at 200 California sensors, and studies whether such forecasters can be trained **federatedly** across four regional data holders instead of pooling their data. Everything runs on historical XTraffic data (2022–2024); nothing is live.

It is one web application with five functional areas:

| Area | User sees | Backed by |
|---|---|---|
| Traffic forecasting | Overview, Forecasting pages | `POST /api/predict` → `server.Engine` (2023 GRU study models) |
| Model evaluation | Performance page | `GET /api/metrics`, `/api/models` (frozen aggregates) |
| Federated learning | Federated, Methodology, About pages | frozen study results; offline training code in `src/aiori`, `src/aiori2` |
| Historical traffic replay | `/replay` | `GET /api/replay/*` → `replay.Replay` (frozen MLP + FedAvgM) |
| Historical disruption analysis | `/disruptions` | `GET /api/replay/disruptions*` → `disruptions.Disruptions` (offline catalog) |

## 2. System boundaries

- **Inside:** React single-page application (`src/app/web`), two standard-library Python HTTP servers (`src/app/backend`), read-only artifact files (`src/data`, `src/results`), offline pipelines (`src/aiori`, `src/aiori2`, `src/app/backend/disruption_detect.py`, `src/experiments`).
- **Outside:** the user's browser; OpenStreetMap tile servers (Replay map background); XTraffic source files (inputs to offline pipelines); Cloudflare quick tunnel (configured, not running).
- **Not present:** no database server, message queue, cache server, cloud runtime, authentication, live data feed or streaming.

## 3. Architecture in one picture

[HLD-01 System context](diagrams/hld-01-system-context/aiori-hld-01-system-context.html) · [HLD-02 Layered platform](diagrams/hld-02-layered-platform/aiori-hld-02-layered-platform.html)

```
Browser ──► React SPA (static bundle) ──► /api/*         ──► server.Handler ─► Engine ─► 2023 GRU checkpoints + nine_month data
                                     └─► /api/replay/* ──► replay_server.Handler ─► Replay ─► 3-year arrays + frozen MLP + FedAvgM
                                                                                 └► Disruptions ─► offline anomaly catalog
Offline (never at request time): aiori2.pipeline → aiori2.run (centralised / local / IID FL / geographic FL) → plan → final_eval → frozen manifest
                                  disruption_detect.py build → versioned catalog
```

Two server processes serve the same application:

| Process | Port | Serves | Status on 2026-11-15 |
|---|---|---|---|
| `server.py` | 8000 | `web/dist` (6 routes) + study API | running (started manually, `AIORI_CANDIDATE=1`) |
| `replay_server.py` | 8010 | `web/dist-replay` (8 routes) + study API + replay/disruption API | running |
| `replay_server.py` | 8020 | Phase 10 redesign preview build | running; isolated, unapproved, not production |

`replay_server.Handler` subclasses `server.Handler`, so :8010 answers every :8000 endpoint as well; they are two processes, not one.

## 4. Layers (summary of the HLD)

1. **Presentation:** eight React pages; Replay and Disruption Explorer exist only in `vite --mode replay` builds.
2. **Frontend clients:** `lib/api.ts` (study API, `StudyProvider`), `lib/replay.ts`, `lib/disruptions.ts`, `useRemote` (abort + bounded LRU cache).
3. **HTTP:** `ThreadingHTTPServer`, JSON responses, strict query validation, uniform `{error:{status,message}}` errors, static files with SPA fallback.
4. **Services:** `Engine` (GRU inference, metrics, demo), `Candidate` (opt-in optimised MLP), `Replay` (frames, series, forecasts), `Disruptions` (catalog queries, series, robustness).
5. **Models:** ForecastGRU (2023 study), MLP + FedAvgM history A (three-year study), optimised MLP candidate; all frozen, loaded once, CPU.
6. **Data:** NumPy arrays (memory-mapped), JSON/CSV results, PyTorch checkpoints, versioned catalog.
7. **Offline:** preprocessing, five-model × four-arm × eight-algorithm federated experiments, validation-driven selection, one-time frozen test, anomaly catalog build.

## 5. Key workflows

| Flow | Runtime or offline | Diagram |
|---|---|---|
| Forecast request (`POST /api/predict`) | runtime | [LLD-03](diagrams/lld-03-forecast-sequence/aiori-lld-03-forecast-sequence.html) |
| Replay map frames and sensor forecast | runtime | [LLD-04](diagrams/lld-04-replay-sequence/aiori-lld-04-replay-sequence.html) |
| Disruption investigation, replay link, forecast comparison | runtime | [LLD-06](diagrams/lld-06-disruption-sequence/aiori-lld-06-disruption-sequence.html) |
| Anomaly catalog build | offline | [LLD-05](diagrams/lld-05-disruption-pipeline/aiori-lld-05-disruption-pipeline.html) |
| Federated training round | offline | [LLD-07](diagrams/lld-07-federated-sequence/aiori-lld-07-federated-sequence.html) |
| Model selection and frozen test | offline | [HLD-04](diagrams/hld-04-ml-federated/aiori-hld-04-ml-federated.html) |

Full step-by-step traces: [AIORI_DATA_FLOW.md](AIORI_DATA_FLOW.md).

## 6. Architecture decisions (as implemented)

| Decision | Reason visible in code and reports |
|---|---|
| Standard-library HTTP server, no web framework | Minimal dependencies; Dockerfile installs only numpy, pandas and CPU torch |
| Models and data loaded once at start-up; read-only at request time | Fast requests (ms-scale inference), no write path, no state to lose |
| Hash checks on frozen checkpoints and catalog before serving | A changed artifact refuses to start (models) or disables only the explorer (catalog) |
| Replay features in a separate process and a separate build | The production entry point and its bundle stay unchanged; rollback = stop the process |
| Federated learning simulated in one process | Same data and seeds for every arm; deterministic resume; no networking needed for the study |
| Validation-only selection; one frozen test evaluation | `final_eval` refuses re-runs and changed manifests; training never imports it |
| Offline, versioned anomaly catalog | No full-dataset scan per request; deterministic event ids; reproducible rebuild |
| Memory-mapped NumPy arrays | 200 × 315,648 arrays without loading everything into RAM |

## 7. Scientific and operational limitations

- No live traffic: all readings are historical XTraffic data; timestamps are naive Pacific time, inferred.
- Sensor coordinates are provisional (row-to-station mapping supported but unconfirmed).
- No congestion levels: only flow is used; speed/occupancy thresholds are not validated.
- Candidate disruption events are statistical departures found retrospectively; causes (accidents, closures, earthquakes, weather) are not known; ~a third of multi-sensor groups could arise by chance.
- No disaster-specific accuracy is claimed. The exploratory anomaly-window comparison (pooled WAPE ~32–33% inside anomalies vs ~10–11% on matched normal days) is not a test score.
- Test WAPE 10.506% (IID) and 10.546% (geographic) are three-seed means over the whole Jul–Dec 2024 test period, not per-forecast guarantees.
- Two model generations are served: the 2023 GRU study on the Forecasting page and the three-year MLP + FedAvgM in Replay and the Explorer.
- Operations: no supervisor is active (keep-alive task not registered, tunnel not running); a process stops if its model checks fail.

## 8. Diagram index

| Id | Title | Type |
|---|---|---|
| HLD-01 | [System context](diagrams/hld-01-system-context/aiori-hld-01-system-context.html) | architecture |
| HLD-02 | [Layered platform](diagrams/hld-02-layered-platform/aiori-hld-02-layered-platform.html) | architecture |
| HLD-03 | [Frontend features and backend capabilities](diagrams/hld-03-frontend-features/aiori-hld-03-frontend-features.html) | architecture |
| HLD-04 | [Machine learning and federated learning](diagrams/hld-04-ml-federated/aiori-hld-04-ml-federated.html) | architecture |
| HLD-05 | [Runtime and deployment](diagrams/hld-05-runtime-deployment/aiori-hld-05-runtime-deployment.html) | architecture |
| LLD-01 | [Frontend component tree](diagrams/lld-01-frontend-tree/aiori-lld-01-frontend-tree.html) | tree |
| LLD-02 | [Backend modules](diagrams/lld-02-backend-modules/aiori-lld-02-backend-modules.html) | architecture |
| LLD-03 | [Historical forecast request](diagrams/lld-03-forecast-sequence/aiori-lld-03-forecast-sequence.html) | sequence |
| LLD-04 | [Replay interaction](diagrams/lld-04-replay-sequence/aiori-lld-04-replay-sequence.html) | sequence |
| LLD-05 | [Disruption detection pipeline](diagrams/lld-05-disruption-pipeline/aiori-lld-05-disruption-pipeline.html) | dataflow |
| LLD-06 | [Disruption investigation](diagrams/lld-06-disruption-sequence/aiori-lld-06-disruption-sequence.html) | sequence |
| LLD-07 | [Federated training round](diagrams/lld-07-federated-sequence/aiori-lld-07-federated-sequence.html) | sequence |
| LLD-08 | [Checkpoint and catalog validation](diagrams/lld-08-checkpoint-validation/aiori-lld-08-checkpoint-validation.html) | workflow |
| LLD-09 | [Data and artifact dependencies](diagrams/lld-09-data-artifacts/aiori-lld-09-data-artifacts.html) | dataflow |
| LLD-10 | [Server start-up and operations](diagrams/lld-10-runtime-startup/aiori-lld-10-runtime-startup.html) | workflow |

Each diagram folder holds its editable source (`candidate.json`), the standalone interactive HTML and Archify's check receipts. The HTML viewer can export PNG, SVG and WebM.
