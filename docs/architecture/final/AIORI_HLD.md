# AIORI High-Level Design

Part of the [final architecture](AIORI_FINAL_ARCHITECTURE.md). Each diagram is a standalone interactive HTML file (zoom, route tracing, light/dark, export). Its editable source is `candidate.json` in the same folder.

## HLD-01 · System context

[Open diagram](diagrams/hld-01-system-context/aiori-hld-01-system-context.html)

- **Actors:** users in a web browser. There are no accounts or roles.
- **AIORI web application:** one React single-page application delivered as a static bundle by the server the user opens.
- **Two server processes on one workstation:**
  - Forecast API, `src/app/backend/server.py`, port 8000.
  - Replay & Disruption API, `src/app/backend/replay_server.py`, port 8010.
- **Local artifact files:** processed arrays, checkpoints, frozen results and the disruption catalog, all under `src/data` and `src/results`.
- **Offline pipelines:** produce those files from XTraffic source data. They are never triggered by the website.
- **External at runtime:**
  - OpenStreetMap tiles, used for the Replay map.
  - A Cloudflare quick tunnel to :8000 is configured but was not running when verified.

## HLD-02 · Layered platform

[Open diagram](diagrams/hld-02-layered-platform/aiori-hld-02-layered-platform.html)

| Layer | Components | Interface to the next layer |
|---|---|---|
| Presentation | Overview, Forecasting, Performance, Federated, Methodology, About; Historical Traffic Replay and Disruption Explorer (replay build only) | React components and hooks |
| Frontend clients | `lib/api.ts` + `StudyProvider`; `lib/replay.ts`, `lib/disruptions.ts`; `useRemote` | `fetch` JSON with `AbortController` |
| HTTP | `server.Handler`; `replay_server.Handler` (subclass) | in-process method calls |
| Services | `Engine`, `Candidate` (opt-in), `Replay`, `Disruptions` | NumPy and PyTorch in memory |
| Artifacts | 2023 study files; candidate v1; three-year arrays + Stage 3 checkpoints; catalog v1 | local file reads, memory maps, hash checks |

## HLD-03 · Frontend features and backend capabilities

[Open diagram](diagrams/hld-03-frontend-features/aiori-hld-03-frontend-features.html)

| Feature (route) | Backend capability |
|---|---|
| Overview (`/`) | study data via `StudyProvider`; `POST /api/predict` for the example forecast |
| Forecasting (`/forecast`) | `POST /api/predict`: historical hour, or 12 user readings |
| Performance, Federated, Methodology, About | study data: `GET /api/health`, `/metrics`, `/sensors`, `/demo`, `/models` |
| Historical Traffic Replay (`/replay`) | `GET /api/replay/meta`, `/sensors`, `/day`, `/frame`, `/series`, `/forecast` |
| Historical Disruption Explorer (`/disruptions`) | `GET /api/replay/disruptions` (list), `/meta`, `/<id>`, `/<id>/series`, `/<id>/compare`, `/<id>/robustness`; also `/api/replay/forecast` and `/sensors` |
| Explorer → Replay | client-side navigation to `/replay?date&time&sensor&event`, with a link back to the event |

**Build variants:**
- `npm run build` produces `web/dist` with six routes. It contains no Replay or Explorer code, because their lazy imports are `null` in production mode.
- `npm run build:replay` produces `web/dist-replay` with eight routes.
- A redesigned UI (Phase 10) exists only as an isolated preview build in `src/app/web-phase10`. It is not part of the final architecture.

## HLD-04 · Machine learning and federated learning (offline)

[Open diagram](diagrams/hld-04-ml-federated/aiori-hld-04-ml-federated.html)

**Data preparation (`aiori2.pipeline`).** The steps are protect → extract → assemble → quality → experiments → manifest → verify.
- The output is `src/data/processed/aiori2_3yr`, about 4.2 GB:
  - flow array of 200 × 315,648 five-minute steps;
  - conservative and outage masks;
  - calendar;
  - per-experiment filled inputs, train-only scalers, time-of-day medians and window indexes.
- Two training histories:
  - A trains on 2023;
  - B trains on 2022–2023.
  - Both validate on Jan–Jun 2024 and test on Jul–Dec 2024.

**Arms (`aiori2.arms`):**

| Arm | Data | Scaler |
|---|---|---|
| Centralised | all 200 sensors | pooled |
| Local | four independent geographic models | client |
| IID federated | four random 50-sensor clients | per client |
| Geographic non-IID federated | four regional 50-sensor clients | per client |

**Models (`aiori2.models.factory`):** MLP, GRU, TCN, PatchTST, N-HiTS. The contract is input `(B, 12, 1)` scaled flow and output `(B, 12)` scaled forecasts. The models have no sensor embeddings.

**Federated simulation (`aiori2.fl`).** One process; four clients; all participate each round.
- Each client runs 100 local SGD steps (batch 128) on its own stream and returns Δi = wi − w.
- One of eight server rules builds the new global model: FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, SCAFFOLD, FedNova, FedAvgM.

**Runner (`aiori2.run`).**
- Training budget: 20 epochs × 2 M sampled windows.
- Validation every epoch; early stopping with patience 3.
- Saves `best.pt` by validation WAPE.

**Plan and selection (`aiori2.plan`):**
- Stage 1: 40 runs.
- Stage 2: 56 new runs.
- Stage 3: 4 finalists × 2 partitions × 3 seeds = 24 runs.
- Winner by validation: **MLP + FedAvgM, history A**.

**Frozen evaluation (`aiori2.final_eval`).** It hashes every file into `final_manifest.json` and tests once.
- Results: IID 10.506% and geographic 10.546% WAPE (three-seed means); persistence 12.495%.

**Serving:** `replay.Replay` loads seed 20261027 for each partition after hash checks.

**Earlier generation, still served.** The 2023 GRU study (`src/aiori`, nine-month data, four arms with FedAvg) produced the Phase 9 checkpoints served by `server.Engine` on :8000. An optimised MLP (candidate v1, `src/experiments/accuracy_optimization`, not federated) can be enabled with `AIORI_CANDIDATE=1`.

## HLD-05 · Runtime and deployment

[Open diagram](diagrams/hld-05-runtime-deployment/aiori-hld-05-runtime-deployment.html)

**Running**
- :8000 `server.py`.
- :8010 `replay_server.py`.
- :8020 `replay_server.py` with the Phase 10 preview build (isolated, unapproved).

**Configured but not active**
- cloudflared quick tunnel.
- keep-alive scheduled task (not registered).

**Present but not a serving component**
- `AIORI2_Stage3_Watchdog` scheduled task: Ready, `auto_resume` false. It guards training runs.

**Details:** [AIORI_DEPLOYMENT_ARCHITECTURE.md](AIORI_DEPLOYMENT_ARCHITECTURE.md).
