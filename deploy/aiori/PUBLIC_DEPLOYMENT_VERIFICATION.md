# AIORI public deployment: verification record

**Checked:** 2026-11-16, against the Quick Tunnel used that day, `https://induced-struct-firefox-table.trycloudflare.com` (address now expired).
That address belongs to this run only; a new one is issued each time the tunnel starts.

**Result:** all automated checks passed. One check remains for you to do by hand: opening the site on a phone over
mobile data.

## Pre-deployment checks (local)

| Check | Result |
|---|---|
| Docker Desktop 29.9.3 (WSL 2), Compose 5.7.0 | healthy |
| Containers | `aiori-backend-1` healthy, `aiori-web-1` up |
| `http://127.0.0.1:8080` and `/api/health` | 200 |
| Frontend API paths | relative (`/api/...`); no API host is built in |
| Listening ports on the host | only `127.0.0.1:8080`. Backend 8010 is internal to Docker. No Docker daemon port (2375/2376) and no 0.0.0.0 listener |
| Backend routes | GET read-only views, plus one `POST /api/predict` capped at 20 kB. No write, upload, training, admin or debug route. Errors return a generic message |
| Existing tunnels | none running; no keep-alive task registered. The unrelated `AIORI2_Stage3_Watchdog` task was left untouched |

## Public pages (HTTPS through Cloudflare)

| Page | Result |
|---|---|
| `/` Overview | loads |
| `/forecast` | loads |
| `/performance` | loads |
| `/federated` | loads |
| `/methodology` | loads |
| `/about` | loads |
| `/replay` | loads; OpenStreetMap tiles over HTTPS; 200 sensors drawn (dark-mode screenshot) |
| `/disruptions` | loads; 16,445 candidate events; labelled "Candidate anomalies · causes not verified" (light-mode screenshot) |
| Unknown paths | return the site, which shows its own not-found page |

**Across all pages:**
- No error banners.
- No requests over plain `http://`, so no mixed content.
- No CORS errors, because every call is same-origin.
- No development-phase labels or AI-tool credits in the rendered text.

## Public API

| Endpoint | Result |
|---|---|
| `GET /api/health` | `status: ok`, CPU, candidate model enabled |
| `GET /api/models`, `/api/sensors`, `/api/metrics`, `/api/demo` | 3 models, 200 sensors, metrics and demo returned |
| `POST /api/predict`, historical | forecasts from the 2023 GRU, centralised GRU and optimized MLP |
| `GET /api/replay/meta`, `sensors`, `day`, `frame` (observed and forecast), `series` | all 200 |
| `GET /api/replay/forecast` | frozen MLP + FedAvgM inference, 0.8 ms |
| `GET /api/replay/disruptions`, `/meta`, `/<id>`, `/<id>/series`, `/<id>/compare`, `/<id>/robustness` | all 200 |

## Security and privacy probes (public URL)

| Probe | Result |
|---|---|
| `/src/app/backend/server.py`, `/app/backend/server.py`, `/.env`, `/.git/config`, `/compose.yaml`, `/data/processed/`, `/results/`, `/sensor_meta_feature.csv` | only the website shell is returned; no file is served |
| Path traversal (`/api/../src/`, `/api/%2e%2e/server.py`, `/assets/..%2f..%2fetc/passwd`, `/api/replay/../../../etc/passwd`) | website shell or 400; nothing leaked |
| `POST /api/predict` with a 30 kB body | 400 "Send a JSON body under 20 kB" |
| `POST /api/predict` with invalid JSON | 400 |
| `POST /api/predict` with sensor `../../etc/passwd` | 404 "Unknown sensor" |
| `DELETE` and `PUT` | 501, not supported |
| Bad query parameters | 400 with a short message; no stack traces |
| Public JavaScript bundles | no tokens, keys, e-mail addresses, local Windows paths or AI-tool names |
| Images | contain no `.npy`, `.npz`, `.pt` or `.csv` files; data is mounted read-only |
| Response headers | `X-Content-Type-Options: nosniff`, `Referrer-Policy`; nginx and Python versions hidden |

## Not verified automatically

- **Phone on mobile data.** This environment cannot reproduce it. Open the public address on a phone with Wi-Fi off
  and check Overview, Replay (tap a sensor to get a forecast), Disruptions and Forecasting.
- **Survival after the starting session closes.** The tunnel was started through Windows Explorer, in its own process outside
  this session, which is the same way as double-clicking `start-aiori.cmd`. Confirm afterwards with
  `status-aiori.ps1`.
- **Load.** No stress test was run, on purpose. The free Quick Tunnel has no uptime guarantee.
