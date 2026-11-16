# AIORI Deployment Architecture

Verified read-only on 2026-11-15 by listing listening ports, process command lines, scheduled tasks and the cloudflared process. Diagrams: [HLD-05](diagrams/hld-05-runtime-deployment/aiori-hld-05-runtime-deployment.html), [LLD-10](diagrams/lld-10-runtime-startup/aiori-lld-10-runtime-startup.html).

## Host

- One Windows 11 workstation, which is the only deployment target in use.
- Python 3.12 serves on CPU.
- A separate GPU environment (`.venv-gpu`) exists for training only.
- There is no cloud runtime.

## Processes and ports

| Port | Command | Frontend served | APIs | Started by | State |
|---|---|---|---|---|---|
| 8000 | `python src\app\backend\server.py` (env `AIORI_CANDIDATE=1`) | `src/app/web/dist` (production build, 6 routes) | study API | manual start | running |
| 8010 | `python src/app/backend/replay_server.py` | `src/app/web/dist-replay` (8 routes) | study API + replay + disruptions | manual start | running |
| 8020 | `python src/app/backend/replay_server.py` with `AIORI_FRONTEND_DIR=…/web-phase10/dist-phase10`, `AIORI_REPLAY_PORT=8020` | Phase 10 redesign preview | same as 8010 | manual start for design review | running; isolated; **not approved, not production** |

- All three processes bind to 127.0.0.1.
- Each loads its own copy of the models into memory at start-up. The :8010 process used about 0.87 GB of memory.
- They share the same read-only files on disk.

## Configured but not active

| Item | Evidence | State |
|---|---|---|
| Cloudflare quick tunnel to :8000 | `.keepalive/keepalive.ps1`; last URL in `.keepalive/tunnel-url.txt` | no `cloudflared` process; the old URL does not answer |
| Keep-alive watchdog | `.keepalive/keepalive.ps1`, `install-task.ps1` | scheduled task not registered |
| Docker image | `src/app/backend/Dockerfile` (serves `web/dist` with artifacts mounted read-only) | recipe only; not running |
| Hugging Face Space | `src/deploy/hf_deploy.py`, `src/deploy/hf-space/` | script only; not verified |
| Vercel (frontend) | `src/app/web/vercel.json` | config only; not verified |

> **Superseded (2026-11-16).** The final deployment is `deploy/aiori` (Docker Compose, served publicly through a
> Cloudflare Quick Tunnel). The items above are legacy files kept for history; `.keepalive/` stays on the local
> machine and is not part of the repository.

## Scheduled tasks

| Task | Purpose | State |
|---|---|---|
| `\AIORI2\AIORI2_Stage3_Watchdog` | checkpoint-aware guard for the Stage 3 training batch; runs every 10 minutes while the user is logged on | Ready; `auto_resume` false, so it cannot start training |

## Start-up, configuration and recovery

- **Start-up order:**
  - `server.py`: `Engine` (models and data) → HTTP server.
  - `replay_server.py`: frontend dir → `Engine` → `Replay` → `Disruptions` → HTTP server.
- **Configuration** is by environment variable only; see LLD-10 for the full list. The defaults are localhost, CPU and the repository's artifact paths.
- **Failure behaviour:**
  - A missing or changed model or data file stops the process at start-up.
  - A missing or stale disruption catalog returns 503 only on the explorer routes.
- **Recovery:** today a stopped process must be started again by hand.
  - The keep-alive script would restart `server.py` and the tunnel, but it is not registered.
  - Its kill pattern also matches `replay_server.py`.
- **Logging:** request lines on stderr. The keep-alive script logs to `.keepalive/keepalive.log`.
- **Model storage:** checkpoints and arrays stay on local disk. The Dockerfile mounts them read-only rather than copying them into the image.

## Not present

No load balancer, reverse proxy, TLS termination (other than the tunnel when it runs), database, queue, container orchestration, CI/CD deployment or metrics collection.
