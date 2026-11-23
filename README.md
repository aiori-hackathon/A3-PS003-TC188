## Aiori-Traffic
Problem Statement PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation

Team Code : A3-PS003-TC188
 
**Institution:** BMS College of Engineering, Bengaluru  
**Team Name:** Cors Enabled

### Members
- **Hardik Mangal**
- **Dhruv Jain**
- **Dr. Monika Puttaramaiah**
#### YouTube video link : https://youtu.be/RmcWK-jka4o
#### Demonstration video link : https://youtu.be/APK435N8Xjo/
#### Live demo : [`Live Demo`](https://induced-struct-firefox-table.trycloudflare.com/)
---

# AIORI

AIORI forecasts freeway traffic flow one hour ahead and tests whether that forecaster can be trained across regions
**without pooling their data**, using federated learning, while counting exactly what the collaboration costs in network
traffic. It uses recorded XTraffic data from 200 California sensors split into four regional clients. Everything it shows
is historical: it is not a live traffic monitor, and its forecasts are not traffic advisories.


## Core idea

- **Four regions, one shared model.** LA/Ventura, Bay Area, Coastal/South Central and Inland/Central each train the same small model on their own 50 sensors.
- **Only model updates travel.** A server combines the updates with **FedAvgM** (FedAvg with server momentum); raw readings never leave a region.
- **Every byte is counted.** The engine logs the exact payload of each round, so accuracy and communication cost are reported together.
- **One frozen model, served honestly.** The winner was chosen on validation data only, hash-locked, tested once, and is what the website runs.

## Objectives

1. Forecast the next 60 minutes of flow (12 five-minute steps) from the last hour of readings.
2. Compare centralised, local-only, IID-federated and geographic (non-IID) federated training on equal terms.
3. Compare eight federated algorithms and five model architectures, and select one without touching the test data.
4. Measure the communication cost of federation alongside its accuracy.
5. Deliver the result as a working, read-only web application with historical replay and disruption analysis.


## Results

**Selected model: residual MLP (103,052 parameters) + FedAvgM, trained on 2023, tested once on Jul–Dec 2024 (124,985,327 targets).**

| Model | Test WAPE (mean ± SD, 3 seeds) |
|---|---|
| MLP + FedAvgM, IID clients | **10.506% ± 0.031** |
| MLP + FedAvgM, geographic clients | **10.546% ± 0.026** |
| N-HiTS + FedAvg baseline, IID / geographic | 10.728% / 10.742% |
| Naive persistence (repeat the last reading) | 12.495% |

- **About 16% lower error than persistence.** FedAvgM beat plain FedAvg on every seed.
- **Communication cost:** 412 KB per model copy, 3.30 MB per round (4 clients, up and down), 2.64 GB for a full 800-round run.
- **Honest comparison:** non-federated references are still 0.08–0.25 points better, though with half the training budget and one seed.
- **Earlier 2023 study:** a compact GRU trained four ways; local-only was best (WAPE 8.675%), and an optimised per-client MLP reached 7.669% on Oct–Dec 2023. These numbers use a different period and are not comparable with the table above.

WAPE is total absolute error divided by total observed flow (lower is better). Full results and hashes:
[`PHASE8C_REPORT.md`](src/results/aiori2/phase8c_stage3/analysis/PHASE8C_REPORT.md) and
[`src/docs/phase9_results.md`](src/docs/phase9_results.md).

## The website

Eight pages from one address: **Overview**, **Forecasting**, **Performance**, **Federated**, **Methodology**, **About**,
**Historical Traffic Replay** (all 200 sensors on a map, 2022–2024, with a one-hour forecast from any moment) and the
**Historical Disruption Explorer** (16,445 statistical anomaly *candidates*, not confirmed incidents). Light and dark
themes; the home page uses a public-domain photograph of a Los Angeles street.

## Tech stack and training

| Layer | Technology |
|---|---|
| Machine learning | Python 3.12, PyTorch 2.14.1, NumPy 2.5, pandas 3.1; own federated engine [`aiori2.fl`](src/aiori2/fl/) with 8 algorithms |
| Backend API | Python standard-library HTTP server, JSON, CPU inference of the frozen models |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS 4, Recharts, Leaflet with OpenStreetMap |
| Deployment | Docker Compose, nginx 1.27, Cloudflare Quick Tunnel |
| Testing | pytest, Vitest |

Training ran on one **NVIDIA RTX PRO 4000 Blackwell (24 GB)** with CUDA 13.0; the website needs no GPU.

| Phase | Device | Runs | Time |
|---|---|---|---|
| 2023 study (GRU, 4 approaches × 3 seeds) | CPU | 12 | 27.3 h |
| 2023 optimised MLP search | GPU | 11 | 1.9 h |
| Stage 1: 5 models × 2 histories × 4 approaches | GPU | 40 | 7.6 h |
| Stage 2: 8 federated algorithms | GPU | 56 + 8 reused | 13.0 h |
| Stage 3: 4 finalists × 2 client splits × 3 seeds | GPU | 24 | 10.2 h |
| Final test, run once | GPU | 15 checkpoints | 45 min |

Federated settings: clients run SGD (lr 0.01, 100 steps × batch 128 per round); the server uses FedAvgM (lr 1.0,
momentum 0.9); 40 rounds per epoch, up to 20 epochs, early stopping with patience 3; masked MAE loss.

## Repository

| Part | Where |
|---|---|
| 2022–2024 federated study | [`src/aiori2/`](src/aiori2/README.md) |
| Final-model evidence (selection, frozen manifest, test) | [`src/results/aiori2/phase8c_stage3/analysis/`](src/results/aiori2/phase8c_stage3/analysis/) |
| Original 2023 study | `src/aiori/`, `src/experiments/`, [`src/docs/`](src/docs/) |
| Backend API, Replay and Disruption services | `src/app/backend/` |
| Website | `src/app/web-phase10/` (original 2023 dashboard: `src/app/web/`) |
| Docker deployment and presentation scripts | [`deploy/aiori/`](deploy/aiori/README.md) |
| Architecture and project graph | [`docs/architecture/`](docs/architecture/final/AIORI_FINAL_ARCHITECTURE.md), [`docs/graphify/final/`](docs/graphify/final/AIORI_GRAPH_README.md) |

## Setup

### Data and models (not in this repository)

XTraffic is licensed CC BY-NC 4.0 and the files are large, so data and checkpoints are not published. To run the site
you need these folders under `src/`, produced by the research pipelines:

- `data/processed/nine_month/`, `results/phase9/`, `results/accuracy_optimization/` for the 2023 Forecasting page;
- `data/processed/aiori2_3yr/`, `results/aiori2/phase8c_stage3/` (the 15 checkpoints in the frozen manifest) and `sensor_meta_feature.csv` for Replay;
- `results/aiori2/phase9b_disruptions/` for the Disruption Explorer (rebuild with `python src/app/backend/disruption_detect.py build`).

To regenerate them see [`src/docs/phase9_reproduction.md`](src/docs/phase9_reproduction.md) and
[`src/aiori2/RUNNING.md`](src/aiori2/RUNNING.md); training needs the raw XTraffic files and a GPU.

### Option 1: Docker (recommended)

Requires Docker Desktop (Windows with WSL 2). From the repository root:

```powershell
cd deploy\aiori
docker compose up -d --build
```

Open http://127.0.0.1:8080. nginx serves the site and forwards `/api/` to the backend container, which has no public
port; the data folders are mounted read-only. Details: [`deploy/aiori/README.md`](deploy/aiori/README.md).

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

Open http://127.0.0.1:8010. `AIORI_CANDIDATE=1` adds the optimised MLP to the Forecasting page.

### Tests

```powershell
.venv\Scripts\python.exe -m pytest src\app\tests src\tests
cd src\app\web-phase10; npm test
```

## Deployment

The live demo link above is the Docker site running on the presenter's workstation, published over HTTPS by a free
Cloudflare Quick Tunnel. It is reachable only while that machine is on and running Docker and the tunnel, and the
address changes if the tunnel restarts. Start, status, stop and recovery steps:
[`deploy/aiori/PRESENTATION_DAY_RUNBOOK.md`](deploy/aiori/PRESENTATION_DAY_RUNBOOK.md).

**Security:** the API is read-only (no upload, training or admin routes), sensors are shown with anonymised ids such as
`LA-01`, both containers run with read-only file systems, and the repository holds no credentials.

## Limitations

- Federation was simulated in one process; communication is counted from the weights exchanged, not measured on a network.
- No differential privacy or secure aggregation yet.
- Local-only models remain slightly better in an unmatched comparison; a matched comparison is the next step.
- One state, 200 sensors, provisional sensor-to-region mapping, historical data only.

## Licence and credits

XTraffic is licensed CC BY-NC 4.0; raw data, processed arrays, checkpoints, predictions and station identifiers are not
included. The home-page photograph, "Palm lined street in Los Angeles, California", is by Carol M. Highsmith (Library of
Congress, public domain).
