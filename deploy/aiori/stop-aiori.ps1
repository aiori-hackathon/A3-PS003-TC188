<#
  AIORI: stop the Cloudflare tunnel started by start-aiori.ps1, then the AIORI containers.

      powershell -ExecutionPolicy Bypass -File "<repo>\deploy\aiori\stop-aiori.ps1"

  Only the AIORI tunnel (by its saved process id) and the two AIORI containers are stopped. Images, data, models
  and results are kept. Options:  -TunnelOnly   stop the tunnel, keep the local site running
#>
param([switch]$TunnelOnly)

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $Here "logs\tunnel.pid"

if (Test-Path $PidFile) {
    $id = Get-Content $PidFile -ErrorAction SilentlyContinue
    $p = $null
    if ($id) { $p = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "cloudflared" } }
    if ($p) { Stop-Process -Id $p.Id; Write-Host "[AIORI] Tunnel stopped (process $($p.Id))." }
    else { Write-Host "[AIORI] The saved tunnel process is not running." }
    Remove-Item -Force $PidFile
} else {
    Write-Host "[AIORI] No tunnel started by start-aiori.ps1 is recorded."
}

if ($TunnelOnly) { exit 0 }

$Docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $Docker) {
    foreach ($c in @("$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe", "C:\Program Files\Docker\Docker\resources\bin\docker.exe")) {
        if (Test-Path $c) { $Docker = $c; break }
    }
}
if (-not $Docker) { Write-Host "[AIORI] docker.exe not found; containers not stopped."; exit 1 }
& $Docker compose -f (Join-Path $Here "compose.yaml") down
Write-Host "[AIORI] Containers stopped. Start again with start-aiori.ps1."
