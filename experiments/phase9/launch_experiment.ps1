# Phase 9 launcher: records the deadline, registers the per-user scheduled task that runs the watchdog
# (at logon and every 5 minutes; no elevation; no system setting changed) and starts it.
#   powershell -NoProfile -File experiments\phase9\launch_experiment.ps1 -DeadlineUtc 2026-10-29T03:38:10Z
param([Parameter(Mandatory = $true)][string]$DeadlineUtc)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent (Split-Path -Parent $here)
$py = Join-Path $root ".venv\Scripts\python.exe"
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"

& $py (Join-Path $here "run_nine_month_experiment.py") --init-only --deadline-utc $DeadlineUtc

$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$(Join-Path $here 'watchdog.py')`"" -WorkingDirectory $root
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Hours 20)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "AIORI_PS003_Phase9" -Action $action -Trigger @($atLogon, $every5) -Settings $settings `
    -Description "AIORI PS-003 Phase 9 watchdog (unregisters itself when the experiment completes)" -Force | Out-Null
Start-ScheduledTask -TaskName "AIORI_PS003_Phase9"
Write-Output "registered and started AIORI_PS003_Phase9"
