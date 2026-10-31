# Phase 9 — reproduction and operation

The nine-month experiment (Jan 1 – Sep 30, 2023) runs unattended. It is a controller with one worker subprocess per task, plus a watchdog started by the per-user scheduled task `AIORI_PS003_Phase9`. The protocol is in `docs/phase9_protocol.md`. Results are written automatically to `docs/phase9_results.md`, `docs/phase9_executive_summary.md` and `docs/phase9_figures/`; local-only outputs go to `results/phase9/`.

## Check progress (read-only)
```powershell
Get-Content results\phase9\state\heartbeat.json                       # updated every 20 s
Get-Content results\phase9\logs\execution.log -Tail 20
Get-Content results\phase9\logs\recovery.log -Tail 20                 # watchdog restarts, retries, orphan handling
Get-ScheduledTask -TaskName AIORI_PS003_Phase9                        # gone once the experiment completes
```
The `stages` and `runs` fields of `results\phase9\state\experiment_status.json` record every task as completed, failed, skipped or incomplete.

## Start from scratch on another machine
1. Python 3.12 venv with `requirements.txt` (torch 2.14.1+cpu); XTraffic `year_2023/2023_p01..p09.npy`, `node_order.npy`, `sensor_meta_feature.csv`; the Phase 1–8 artifacts under `data/processed/` (January audit, metadata alignment, geo200 selection, pilot manifests).
2. Tests: `.venv\Scripts\python.exe -m pytest -q` (240 tests, synthetic data only).
3. Optional rehearsal on synthetic data, no real data touched: `.venv\Scripts\python.exe experiments\phase9\rehearsal.py <empty work dir> --deadline-minutes 30`.
4. Launch, with the deadline 16 hours ahead: `powershell -NoProfile -File experiments\phase9\launch_experiment.ps1 -DeadlineUtc <YYYY-MM-DDTHH:MM:SSZ>`.
   It registers the task (at logon and every 5 min; no elevation; no system setting changed) and starts the watchdog. The watchdog starts the controller, which runs: audit → prepare → verify → smoke → benchmark → train → freeze → precheck → test → analysis → report.

## Stop early (keeps every checkpoint; a later launch resumes)
```powershell
Unregister-ScheduledTask -TaskName AIORI_PS003_Phase9 -Confirm:$false
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'phase9\\(watchdog|run_nine_month_experiment|worker)' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Recovery behaviour (each part tested)
- **Crashed or killed controller:** the watchdog restarts it within about 60 s. Orphaned training workers are stopped, and every run resumes exactly from its last per-epoch or per-round checkpoint. This was tested on the real run on 2026-10-28 (12:11Z) and in `tests/test_phase9_resumable.py`.
- **Crashed watchdog:** the scheduled task restarts it within 5 minutes; a reboot or logon restarts it at logon.
- **Failed task:** at most 4 attempts, with backoff of 30, 120 and 300 s. An integrity failure (exit code 3) is not retried. At most 6 controller restarts per hour and 40 in total.
- **Deadline:** no run starts unless it is expected to finish before the training cutoff (deadline − 90 min). Runs still going at the cutoff are stopped and excluded. Freeze, pre-check, test (once), analysis and reports always run.
