# AIORI Cloudflare deployment report

**Status:** AIORI is publicly reachable over HTTPS (2026-11-16).

| Item | Value |
|---|---|
| Deployment mode | Docker Compose (two containers) + Cloudflare Quick Tunnel |
| Public URL on 2026-11-16 (historical, expired) | `https://induced-struct-firefox-table.trycloudflare.com`. It changes every time the tunnel restarts; the current one is always in `logs/public-url.txt` |
| Tunnel target | `http://127.0.0.1:8080` only |
| Hosting | the site runs on the presenter's workstation; Cloudflare only provides the HTTPS address and transport |
| Cloudflare client | cloudflared 2026.10.5, already installed in Program Files; nothing new installed |
| Cost | none: free Quick Tunnel, no account, domain or payment method |
| Docker | Desktop 29.9.3 on WSL 2, Compose 5.7.0 |

## Architecture

```
phone / laptop ──HTTPS──► Cloudflare edge ──► cloudflared (this PC) ──► 127.0.0.1:8080 web (nginx)
                                                                         ├─ website
                                                                         └─ /api/* ──► backend:8010 (internal only)
```

## Containers (measured)

| Container | Status | Memory | Published |
|---|---|---|---|
| `aiori-web-1` (nginx, unprivileged, read-only filesystem) | up | 23 MB of 256 MB | `127.0.0.1:8080` |
| `aiori-backend-1` (replay server, CPU) | healthy | 782 MB of 6 GB | none |

## What was done

1. Checked Docker health, the containers, local site and API, relative API paths, host listening ports and every
   backend route (section 2 of `PUBLIC_DEPLOYMENT_VERIFICATION.md`). No security issue was found.
2. Confirmed that no tunnel or keep-alive task was running, and that no cloudflared configuration exists in the user
   profile, so a quick tunnel cannot pick up a retired named-tunnel config.
3. Added operating scripts in this folder:
   - `start-aiori.cmd` and `start-aiori.ps1`: start Docker, the site and the tunnel, then print the URL;
   - `status-aiori.ps1`: read-only health report;
   - `stop-aiori.ps1`: stop the tunnel and containers;
   - `.gitignore` for `logs/`.
4. Started the tunnel with `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080 --logfile logs\tunnel.log`
   through `start-aiori.cmd`, opened by Windows Explorer, so it runs in its own minimised window, independent of
   the terminal session that started it.
5. Verified the public site, APIs and security probes (see `PUBLIC_DEPLOYMENT_VERIFICATION.md`).

A first attempt to launch the script detached through WMI did not start (likely blocked by security software). It
was not retried and no security setting was changed.

## Unchanged

- AIORI source code, frontend, backend, API contracts, Docker images and the compose file.
- Models, data, checkpoints and results.
- The original services on ports 8000, 8010 and 8020: not running and not touched.
- The `AIORI2_Stage3_Watchdog` scheduled task.
- No firewall or router change. No Windows service or scheduled task was created.
- Nothing committed or pushed.

## Limitations

- **The Quick Tunnel address is temporary.** It changes after every restart, and Cloudflare gives no uptime
  guarantee for quick tunnels. A permanent address would need a named tunnel and a domain in a Cloudflare account,
  which needs your approval.
- **The site is up only while this PC is on, awake, online, and running Docker and the cloudflared window.**
- **The containers restart with Docker Desktop, but the tunnel does not.** Run `start-aiori.cmd` after a reboot.
- **The API still returns internal model ids** such as `aiori2-mlp-fedavgm-…` in JSON. The website shows them as
  "AIORI MLP + FedAvgM …"; the raw ids appear only to someone reading the API directly.
- **The phone-on-mobile-data check is manual** (see the runbook).

## Commands

| Action | Command |
|---|---|
| Start | `start-aiori.cmd` in `deploy\aiori` (double-click), or `& "$AIORI\start-aiori.cmd"` |
| Status | `powershell -ExecutionPolicy Bypass -File "$AIORI\status-aiori.ps1"` |
| Stop | `powershell -ExecutionPolicy Bypass -File "$AIORI\stop-aiori.ps1"` (`-TunnelOnly` keeps the local site) |

Full procedure: `PRESENTATION_DAY_RUNBOOK.md`.
