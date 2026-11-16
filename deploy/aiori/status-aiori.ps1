<#
  AIORI: show container status, local and public health, and the current public URL. Changes nothing.

      powershell -ExecutionPolicy Bypass -File "<repo>\deploy\aiori\status-aiori.ps1"
#>
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Logs = Join-Path $Here "logs"

function Check($label, $url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10
        Write-Host ("  {0,-22} OK ({1})" -f $label, $r.StatusCode)
    } catch { Write-Host ("  {0,-22} NOT RESPONDING" -f $label) }
}

$Docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $Docker) {
    foreach ($c in @("$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe", "C:\Program Files\Docker\Docker\resources\bin\docker.exe")) {
        if (Test-Path $c) { $Docker = $c; break }
    }
}
Write-Host "Containers:"
if ($Docker) { & $Docker compose -f (Join-Path $Here "compose.yaml") ps --format "  {{.Name}}  {{.Status}}" } else { Write-Host "  docker.exe not found" }

Write-Host "Local:"
Check "site" "http://127.0.0.1:8080/"
Check "API health" "http://127.0.0.1:8080/api/health"

Write-Host "Tunnel:"
$pidFile = Join-Path $Logs "tunnel.pid"
$p = $null
if (Test-Path $pidFile) { $p = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "cloudflared" } }
if ($p) { Write-Host "  cloudflared running (process $($p.Id), started $($p.StartTime))" } else { Write-Host "  not running" }

$urlFile = Join-Path $Logs "public-url.txt"
if (Test-Path $urlFile) {
    $url = (Get-Content $urlFile | Select-Object -First 1).Trim()
    Write-Host "  public URL: $url"
    if ($p) { Check "public site" "$url/"; Check "public API health" "$url/api/health" }
} else { Write-Host "  no public URL recorded" }
