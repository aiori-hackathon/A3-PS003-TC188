## AIORI-3-HACKATHON-PROJECTS
### Team Name : Cors Enabled
### Team Code : A3-PS003-TC188
### PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation
#### YouTube video link : https://youtu.be/RmcWK-jka4o
#### Demonstration video link : https://youtu.be/APK435N8Xjo/
#### Live demo : shown during the presentation from a temporary Cloudflare Quick Tunnel address (see [Deployment](#deployment)). There is no permanent public URL.

---

# AIORI

AIORI forecasts freeway traffic flow one hour ahead and studies whether such a forecaster can be trained across regions
without pooling their data, using federated learning. It works on recorded XTraffic data from 200 California sensors,
grouped into four regional clients. Everything AIORI shows is **historical**: it is not a live traffic monitor, and its
forecasts are not traffic advisories.

Problem statement: [`AIORI3_A3-PS003-TC188.pdf`](AIORI3_A3-PS003-TC188.pdf).

## What is in this repository

| Part | Where |
|---|---|
| Original 2023 study: compact GRU, four training approaches, optimised MLP | `src/aiori/`, `src/experiments/`, [`src/docs/`](src/docs/phase9_results.md) |
| 2022–2024 federated study: five architectures, eight federated algorithms, three-stage selection | [`src/aiori2/`](src/aiori2/README.md) |
| Final-model evidence (selection, frozen manifest, one-time test) | [`src/results/aiori2/phase8c_stage3/analysis/`](src/results/aiori2/phase8c_stage3/analysis/PHASE8C_REPORT.md) |
| Backend: study API, Historical Traffic Replay, Historical Disruption Explorer | `src/app/backend/` |
| Final AIORI website | `src/app/web-phase10/` |
| Original dashboard (2023 study) | `src/app/web/` |
| Docker deployment and presentation scripts | [`deploy/aiori/`](deploy/aiori/README.md) |
| Architecture and project graph | [`docs/architecture/`](docs/architecture/final/AIORI_FINAL_ARCHITECTURE.md), [`docs/graphify/final/`](docs/graphify/final/AIORI_GRAPH_README.md) |
| Tests | `src/tests/`, `src/app/tests/`, `*.test.ts` in both frontends |

## Research

### Original 2023 study

- **Setup:** 200 sensors and four geographic clients (50 sensors each).
- **Data split:** XTraffic 2023, Jan–Sep. Train Jan 1 – Jul 10, validation Jul 11 – Aug 7, held-out test Aug 8 – Sep 30.
- **Model:** a compact GRU, 12 past steps in and 12 steps out (60 minutes).
- **Four training approaches:**
  - centralised: all data pooled;
  - local-only: each client trains alone;
  - IID FedAvg;
  - geographic FedAvg.

**Held-out test, mean of 3 seeds:**

| Approach | MAE (vehicles / 5 min) | WAPE |
|---|---|---|
| Local-only GRU (study final model) | 16.990 | 8.675% (8.672% clipped at 0) |
| Centralised GRU | 17.134 | 8.749% |
| IID FedAvg | 17.274 | 8.820% |
| Geographic FedAvg | 17.362 | 8.865% |
| Naive persistence | 24.421 | 12.470% |

**Optimised MLP:** a later residual MLP with a one-day context, trained per client and not federated.
- WAPE 7.669% on an independent Oct–Dec 2023 evaluation, against 8.834% for the GRU on the same windows.
- Details: [`src/docs/phase9_results.md`](src/docs/phase9_results.md) and [`src/docs/accuracy_optimization_report.md`](src/docs/accuracy_optimization_report.md).

### 2022–2024 federated study

- **Data and split:** three years of 5-minute flow for the same 200 sensors and four clients. Training uses 2023 (history A) or 2022–2023 (history B); validation is Jan–Jun 2024; the test is Jul–Dec 2024.
- **Five architectures:** MLP, GRU, TCN, PatchTST, N-HiTS.
- **Four training approaches:** centralised, local-only, IID federated, geographic (non-IID) federated.
- **Eight federated algorithms:** FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, SCAFFOLD, FedNova, FedAvgM.
- **Staged selection on validation data only:**
  - Stage 1: 40 runs (5 architectures × 2 histories × 4 approaches);
  - Stage 2: 64 runs (MLP and N-HiTS × 2 histories × 2 client splits × 8 algorithms; 8 reused from Stage 1);
  - Stage 3: 24 runs (4 finalists × 2 splits × 3 seeds, 20-epoch budget).
- **Selection and test:** the winner was frozen with SHA-256 hashes of every checkpoint, the data and the evaluation code, then tested once.

**Selected model: MLP + FedAvgM (server momentum 0.9), history A.**

| Test, Jul–Dec 2024 (124,985,327 targets) | WAPE, mean ± SD of 3 seeds |
|---|---|
| MLP + FedAvgM, IID clients | **10.506% ± 0.031** |
| MLP + FedAvgM, geographic clients | **10.546% ± 0.026** |
| N-HiTS + FedAvg baseline, IID / geographic | 10.728% / 10.742% |
| Naive persistence | 12.495% |

- **FedAvgM vs plain FedAvg:** FedAvgM beat the FedAvg baseline for every seed, by 0.22 (IID) and 0.20 (geographic) WAPE points.
- **Federated vs non-federated:** the non-federated references are 0.08–0.25 points better (local-only GRU 10.299%, local-only MLP 10.368%, centralised MLP 10.424%). They were trained with half the budget and a single seed, so this comparison is not matched.
- **Not comparable with 2023:** the 2023 and 2022–2024 numbers use different periods and target sets.
- **More:** full results, limitations and hashes are in [`PHASE8C_REPORT.md`](src/results/aiori2/phase8c_stage3/analysis/PHASE8C_REPORT.md).

WAPE is total absolute error divided by total observed flow. Lower is better, and it is not an accuracy percentage.

## The AIORI website

Eight pages, served from one origin:

| Page | What it shows |
|---|---|
| Overview | The project and its main results |
| Forecasting | 2023 study models on historical windows, or a forecast from 12 readings you enter |
| Performance | Test results of both studies |
| Federated | How federated training works here, the algorithm and architecture comparisons, and communication cost |
| Methodology | Data, splits, training settings, staged selection, limitations |
| About | Models served, API, how to run |
| Historical Traffic Replay | A map of all 200 sensors, 2022–2024. Step through recorded flow and ask the frozen MLP + FedAvgM for a one-hour forecast from any moment. Sensor locations are provisional |
| Historical Disruption Explorer | 16,445 **candidate** anomalies found offline in the recorded flow. They are statistical candidates; their causes are not verified and they are not confirmed incidents. Details: [`src/app/PHASE9B_DISRUPTION_REPORT.md`](src/app/PHASE9B_DISRUPTION_REPORT.md) |

## Architecture

- [System architecture (interactive HTML)](docs/architecture/AIORI_SYSTEM_ARCHITECTURE.html): download it and open it in a browser, because GitHub shows HTML as source.
- [Final architecture: HLD and LLD](docs/architecture/final/AIORI_FINAL_ARCHITECTURE.md), with 15 diagrams in `docs/architecture/final/diagrams/`.
- [AIORI project graph](docs/graphify/final/AIORI_FINAL_GRAPH.html) and its [guide](docs/graphify/final/AIORI_GRAPH_README.md).

**How it runs:**
- The browser loads the static site and calls same-origin `/api/...` routes.
- A Python standard-library server (`replay_server.py`) answers both the study API and `/api/replay/*` with CPU PyTorch inference.
- It only reads frozen files. Training happens offline and never on request.

## Running AIORI

### What you need that is not in this repository

Data and model files are **not** published: XTraffic is licensed CC BY-NC 4.0, and the files are large. To run the site
you need these local folders under `src/`, produced by the research pipelines:

| Needed for | Path | Size |
|---|---|---|
| Forecasting page (2023 models) | `data/processed/nine_month/`, `results/phase9/` (runs and final), `results/accuracy_optimization/` | ~0.5 GB read |
| Replay and Disruptions | `data/processed/aiori2_3yr/`, `results/aiori2/phase8c_stage3/` (the 15 checkpoints listed in the frozen manifest), `sensor_meta_feature.csv` | ~0.5 GB read |
| Disruption Explorer | `results/aiori2/phase9b_disruptions/` (rebuildable with `python src/app/backend/disruption_detect.py build`) | 15 MB |

**Regenerating:**
- 2023 study artifacts: [`src/docs/phase9_reproduction.md`](src/docs/phase9_reproduction.md).
- Three-year arrays and federated runs: [`src/aiori2/README.md`](src/aiori2/README.md) and [`src/aiori2/RUNNING.md`](src/aiori2/RUNNING.md). They need the raw XTraffic files and a GPU for training.

### Option 1: Docker (recommended)

Requires Docker Desktop (on Windows with WSL 2). From the repository root:

```powershell
cd deploy\aiori
docker compose up -d --build
```

Open http://127.0.0.1:8080.
- **Containers:** an nginx container serves the site and forwards `/api/` to a backend container that is not published on any host port.
- **Data:** the folders above are mounted read-only.
- **More:** [`deploy/aiori/README.md`](deploy/aiori/README.md).

### Option 2: Python services (Windows)

Requires Python 3.12 and Node.js 22 or newer.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==2.5.4 pandas==3.1.0 pytest==9.1.1
.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
cd src\app\web-phase10; npm ci; npm run build:phase10; cd ..\..\..
$env:AIORI_FRONTEND_DIR = "src\app\web-phase10\dist-phase10"
$env:AIORI_CANDIDATE = "1"
.venv\Scripts\python.exe src\app\backend\replay_server.py
```

Open http://127.0.0.1:8010.

`AIORI_CANDIDATE=1` adds the optimised MLP to the Forecasting page. `src\app\backend\server.py` alone (port 8000) serves
only the 2023 study API.

### Tests

```powershell
.venv\Scripts\python.exe -m pytest src\app\tests src\tests
cd src\app\web-phase10; npm test
```

## Deployment

For the presentation, the Docker site is made public with a free **Cloudflare Quick Tunnel**:
- **Where it runs:** the tunnel forwards HTTPS requests to the site running on the presenter's workstation. Cloudflare provides the address and transport, not the hosting.
- **Address:** it is temporary and changes every time the tunnel restarts, so this README links none.
- **Availability:** the site is reachable only while that PC is on, online and running Docker and the tunnel.
- **Start, check, stop:** [`deploy/aiori/PRESENTATION_DAY_RUNBOOK.md`](deploy/aiori/PRESENTATION_DAY_RUNBOOK.md) covers start (`start-aiori.cmd`), status, stop and recovery.

## Security and privacy

- **Read-only API:** no write, upload, training or admin route. `POST /api/predict` is capped at 20 kB, and errors return a short generic message.
- **Anonymised sensors:** the API and site use ids such as `LA-01`, never station ids or file paths.
- **Docker:** the backend is reachable only from the web container. Both containers have read-only file systems, and the data mounts are read-only.
- **No secrets:** the repository holds no credentials, and the Quick Tunnel needs none.

## Limitations

- **Historical data only:** sensor locations and client regions come from a provisional mapping.
- **Simulated federation:** federated learning ran in one process. Communication is counted from the weights exchanged, not measured on a network.
- **Final model:**
  - three seeds per configuration;
  - most Stage 3 runs were still improving at the 20-epoch cap;
  - in Stage 3 the model and the training history are confounded.
- **Disruptions are candidates:** statistical candidates, not verified incidents.
- **Quick Tunnel availability:** Cloudflare gives no uptime guarantee for Quick Tunnels.

## Data licence

XTraffic is licensed CC BY-NC 4.0. Raw data, processed arrays, model checkpoints, predictions and station identifiers are
not included in this repository.
