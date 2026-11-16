# AIORI: local Docker deployment

**Status: built and tested locally (2026-11-16).** Both images build, the backend is healthy about 10 seconds after
start, and every page and API route answers through http://127.0.0.1:8080 (see "Verified" below). Nothing is published
beyond this machine. The setup does not change the existing Dockerfiles, backend code, models, data or results.

## What runs

```
browser ──► http://127.0.0.1:8080 ──► web (nginx)
                                        ├─ /            final AIORI website (static build, client-side routes)
                                        └─ /api/*  ──►  backend:8010 (internal network only)
                                                         replay_server.py: study API, Historical Traffic Replay,
                                                         Historical Disruption Explorer, frozen-model inference
                                                         reads mounted files, read-only
```

| Service | Image | Built from | Port |
|---|---|---|---|
| `web` | `aiori-web:local` (nginx, unprivileged) | `src/app/web-phase10`, built inside Docker with `vite build --mode replay` | `127.0.0.1:8080` on the host, the only published port |
| `backend` | `aiori-backend:local` (Python 3.12, CPU PyTorch) | `src/aiori`, `src/aiori2`, `src/rebuild_aiori_metadata.py` (client constants), `src/experiments/accuracy_optimization/aopt`, `src/app/backend` | `8010`, internal network only |

The API stays same-origin: the frontend calls relative `/api/...` URLs and nginx forwards them, so no CORS setting and
no API host is needed.

## Read-only mounts

Container paths mirror the repository's `src/` folder (`/srv/src/...`), so the backend finds every file with its
default paths. No environment overrides, and no code changes, are needed for the data.

| Host path (repository) | Container path | Size on disk | Read at run time | Used by |
|---|---|---|---|---|
| `src/data/processed/nine_month` | `/srv/src/data/processed/nine_month` | 450 MB | most of it | 2023 study API (Forecasting page) |
| `src/data/processed/aiori2_3yr` | `/srv/src/data/processed/aiori2_3yr` | 4.2 GB | about 0.5 GB (memory-mapped) | Replay, Disruption Explorer |
| `src/results/phase9` | `/srv/src/results/phase9` | 4.8 GB | a few MB (two checkpoints and JSON) | 2023 study models and metrics |
| `src/results/accuracy_optimization` | `/srv/src/results/accuracy_optimization` | 0.9 GB | about 40 MB | optimized MLP on the Forecasting page |
| `src/results/aiori2/phase8c_stage3` | `/srv/src/results/aiori2/phase8c_stage3` | 41 MB | frozen manifest and checkpoints | frozen MLP + FedAvgM; every hash is checked at start-up |
| `src/results/aiori2/phase9b_disruptions` | `/srv/src/results/aiori2/phase9b_disruptions` | 15 MB | all | disruption catalog |
| `src/sensor_meta_feature.csv` | `/srv/src/sensor_meta_feature.csv` | 3.6 MB | all | sensor positions for the map |

All mounts are `:ro`, and the backend container's own filesystem is read-only too: nothing can be written to the
data, checkpoints or results. Nothing from these folders is copied into an image (each Dockerfile has its own
allow-list `.dockerignore`).

## Docker Desktop on Windows (WSL 2)

Checked on this machine: Docker Desktop 29.9.3 with the WSL 2 backend (`docker-desktop` distribution running, kernel
6.18 WSL2), Compose v5.7.0, 24 CPUs and 31 GB available to the Docker VM. `docker compose config` resolves every mount
to its Windows source path (for example `<repo>\src\data\processed\nine_month` →
`/srv/src/data/processed/nine_month`). Docker Desktop shares the C: drive with the WSL 2 VM, so no extra file-sharing
setting is needed.

Reading through a Windows bind mount is slower than reading a Linux disk. Expect a slower first start than the local
Python service while the checkpoints are hashed and the arrays mapped. Requests afterwards are served from memory.

## Resources (measured)

| | Measured | Notes |
|---|---|---|
| Backend image | 1.56 GB | Python 3.12 slim + CPU PyTorch 2.14.1 + NumPy + pandas; about 0.8 MB of AIORI source |
| Web image | 76 MB | unprivileged nginx + the built site (1.3 MB) |
| Build cache | 2.1 GB | Node build stage and pip downloads; removable with `docker builder prune` |
| Backend memory | 769 MB after loading every page | capped at 6 GB |
| Web memory | 23 MB | capped at 256 MB |
| Backend start-up | about 10 s to healthy | replay data 0.7 s, disruption catalog 2.5 s |

## Verified

- **Pages:** `/`, `/forecast`, `/performance`, `/federated`, `/methodology`, `/about`, `/replay`, `/disruptions` and
  unknown paths return the site; built assets are served with long-lived caching.
- **Study API:** health, models, sensors, metrics, demo, and `POST /api/predict` with all three models (2023 GRU,
  centralised GRU, optimized MLP) for a historical time and for 12 user readings.
- **Replay:** meta, sensors, day, observed and forecast frames, series, and a frozen MLP + FedAvgM forecast (under 1 ms).
- **Disruption Explorer:** meta, the 16,445-event list, and an event with its series, comparison and robustness views.
- **Errors:** bad input returns 400 and unknown API paths 404.
- **Isolation:** port 8010 is not reachable from the host. The images contain no `.npy`, `.npz`, `.pt` or `.csv` files.
- **Labels:** the rendered pages contain no development-phase labels, internal model ids or AI-tool credits.

## Run

For public access with the Cloudflare Quick Tunnel, use `start-aiori.cmd`, `status-aiori.ps1` and `stop-aiori.ps1`
(see `PRESENTATION_DAY_RUNBOOK.md`). Local only:

```
cd deploy/aiori
docker compose up -d --build
docker compose ps                     # backend becomes "healthy" once data and models are loaded
```

Open http://127.0.0.1:8080. Stop with `docker compose down`. That removes the containers only. The images stay until
`docker image rm aiori-web:local aiori-backend:local`.

## Not included

- No public exposure. Port 8080 is bound to `127.0.0.1`. A tunnel is a separate, later step, and only after the site
  works end to end through Docker.
- No training. The containers only read frozen artifacts.
- No change to the existing services on ports 8000, 8010 or 8020, the old `src/app/backend/Dockerfile`, or the old
  Hugging Face Dockerfile.
