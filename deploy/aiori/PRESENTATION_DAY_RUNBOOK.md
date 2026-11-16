# AIORI presentation-day runbook

Use this to run the public AIORI website on its own, with no development tools running. Everything runs on this Windows PC: Docker Desktop
serves the site at `http://127.0.0.1:8080`, and a free Cloudflare Quick Tunnel makes it public over HTTPS.

> **The public address changes every time the tunnel starts.** It looks like `https://<random-words>.trycloudflare.com`.
> After any restart (of the tunnel, the PC or Docker), get the new address from step 2 and share that one.
> Old addresses stop working.

## 0. Set the folder once

In the PowerShell window you use, point `$AIORI` at the `deploy\aiori` folder of your copy of the repository:

```
$AIORI = "C:\path\to\your\clone\deploy\aiori"
```

All commands below use it. The scripts find their own folder (`$MyInvocation` in PowerShell, `%~dp0` in the `.cmd`), so they work from any clone location.

## 1. Start everything (one step)

Double-click:

```
& "$AIORI\start-aiori.cmd"
```

(or double-click `start-aiori.cmd` in Explorer)

or run it in PowerShell:

```
powershell -ExecutionPolicy Bypass -File "$AIORI\start-aiori.ps1"
```

The script:
1. starts Docker Desktop if needed (1–3 minutes);
2. starts the two AIORI containers;
3. waits until the site and API answer locally;
4. starts the tunnel in a minimised **cloudflared** window;
5. prints the public address and checks that it works.

The script is safe to run again: running containers are kept, and a tunnel it already started is reused.

**Leave the minimised cloudflared window open.** Closing it closes the public site. The start window itself can be
closed.

## 2. Find the current public address

- **File:** `$AIORI\logs\public-url.txt`
- **Command:** `Get-Content "$AIORI\logs\public-url.txt"`
- **If both fail:** search the tunnel log for `trycloudflare.com`:
  `Select-String -Path "$AIORI\logs\tunnel.log" -Pattern "trycloudflare.com"`

## 3. Check health

```
powershell -ExecutionPolicy Bypass -File "$AIORI\status-aiori.ps1"
```

This shows container status, the local site and API, whether the tunnel is running, the public address, and whether
it answers. It changes nothing.

**What healthy looks like:**
- `aiori-backend-1 ... (healthy)` and `aiori-web-1 Up ...`;
- every check shows `OK (200)`.

## 4. Logs

| What | Command |
|---|---|
| Backend | `docker compose -f "$AIORI\compose.yaml" logs --tail 50 backend` |
| Website (nginx) | `docker compose -f "$AIORI\compose.yaml" logs --tail 50 web` |
| Tunnel | `Get-Content "$AIORI\logs\tunnel.log" -Tail 30` |

If `docker` is "not recognized", use the full path: `& "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"`.

## 5. Stop

| Goal | Command |
|---|---|
| Stop the tunnel and the containers | `powershell -ExecutionPolicy Bypass -File "$AIORI\stop-aiori.ps1"` |
| Stop only the public tunnel (the site keeps running on `127.0.0.1:8080`) | add `-TunnelOnly` to the line above |

This stops only the AIORI tunnel (by its saved process id) and the two AIORI containers. Images, data, models and
results are kept.

## 6. Restart after a problem

| Symptom | Fix |
|---|---|
| Public address shows a Cloudflare error (1033, 530, 502) | The tunnel stopped or lost its connection. Run `stop-aiori.ps1 -TunnelOnly`, then `start-aiori.cmd`. **Share the new address.** |
| Public address works but pages show "service unavailable" | The backend is down. Run `start-aiori.cmd`: it restarts the containers and reuses the tunnel. |
| `start-aiori` says Docker did not become ready | Open Docker Desktop from the Start menu and wait for "Engine running", then run `start-aiori.cmd` again. |
| `start-aiori` says the site did not answer | Check the backend log (section 4). A message about a missing file or a hash mismatch means a data folder under `src\data` or `src\results` was moved or changed. Restore it; do not edit the models. |
| Port 8080 already in use | Another program took the port. Close it, or run `stop-aiori.ps1`, then `start-aiori.cmd`. |
| No address after 60 seconds | Check the internet connection and the tunnel log. Cloudflare's free quick tunnels are occasionally unavailable; wait a minute and try again. |

The containers restart on their own when Docker Desktop starts (`restart: unless-stopped`), but the tunnel does not.

## 7. After a Windows restart

1. Sign in to Windows.
2. Double-click `start-aiori.cmd`.
3. Share the **new** public address it prints.

Nothing is registered to start automatically.

## 8. Before the presentation (November 26)

- [ ] The day before, run `start-aiori.cmd` and open the printed address on your phone **using mobile data, not
      Wi-Fi**. Check the Overview, Replay (map and a sensor forecast), Disruptions and Forecasting pages.
- [ ] On the day, start it at least 15 minutes early and put the new address into your slides or chat.
- [ ] Keep the PC awake and plugged in: set sleep to "Never" while presenting. Sleep stops the public site.
- [ ] Keep the PC online. If the network drops, restart the tunnel (section 6) and share the new address.
- [ ] Keep a fallback: the same site runs locally at `http://127.0.0.1:8080` even without internet.

## 9. What not to do

- Don't run `docker system prune`, `docker volume rm` or `git clean`. They are not needed and could remove
  working files.
- Don't edit files under `src\data` or `src\results`. The backend checks the frozen model hashes and will refuse to
  start if they change.
- Don't share the backend port. Only the tunnel address (or `127.0.0.1:8080` locally) is meant to be used.
