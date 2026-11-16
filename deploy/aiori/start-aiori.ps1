<#
  AIORI: start the Docker site and a free Cloudflare Quick Tunnel, then print the public URL.

  Run from any PowerShell window:
      powershell -ExecutionPolicy Bypass -File "<repo>\deploy\aiori\start-aiori.ps1"

  Safe to run again: running containers are left as they are, and a tunnel that is already running is reused.
  It only starts things. It never deletes data, never changes models, and registers no service or scheduled task.
  Options:  -NoTunnel   start the local site only (http://127.0.0.1:8080)
#>
param([switch]$NoTunnel)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Logs = Join-Path $Here "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$PidFile = Join-Path $Logs "tunnel.pid"
$UrlFile = Join-Path $Logs "public-url.txt"
$TunnelLog = Join-Path $Logs "tunnel.log"
$Local = "http://127.0.0.1:8080"

function Say($msg) { Write-Host "[AIORI] $msg" }

function Find-Exe($name, $candidates) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

function Test-Url($url, $seconds) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec $seconds
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# 1. Docker Desktop
$Docker = Find-Exe "docker" @(
    "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe")
if (-not $Docker) { throw "docker.exe not found. Install or repair Docker Desktop." }

& $Docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $Desktop = Find-Exe "Docker Desktop.exe" @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "C:\Program Files\Docker\Docker\Docker Desktop.exe")
    if (-not $Desktop) { throw "Docker Desktop is not running and its program was not found. Start Docker Desktop manually." }
    Say "Starting Docker Desktop (this can take 1-3 minutes)..."
    Start-Process -FilePath $Desktop
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        & $Docker info *> $null
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    if (-not $ok) { throw "Docker Desktop did not become ready in 3 minutes. Open Docker Desktop and check it shows 'Engine running'." }
}
Say "Docker is running."

# 2. AIORI containers
Say "Starting the AIORI containers..."
& $Docker compose -f (Join-Path $Here "compose.yaml") up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed. See: docker compose -f `"$Here\compose.yaml`" logs" }

Say "Waiting for the site and API at $Local ..."
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    if ((Test-Url "$Local/api/health" 5) -and (Test-Url "$Local/" 5)) { $ok = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ok) { throw "The site did not answer at $Local within 3 minutes. Check: docker compose -f `"$Here\compose.yaml`" logs backend" }
Say "Local site is healthy: $Local"

if ($NoTunnel) { Say "Tunnel not started (-NoTunnel)."; exit 0 }

# 3. Cloudflare Quick Tunnel (reuse the one this script started, if it is still running)
$Cloudflared = Find-Exe "cloudflared" @(
    "C:\Program Files (x86)\cloudflared\cloudflared.exe",
    "C:\Program Files\cloudflared\cloudflared.exe")
if (-not $Cloudflared) { throw "cloudflared not found. Install it with: winget install --id Cloudflare.cloudflared" }

$running = $null
if (Test-Path $PidFile) {
    $old = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($old) { $running = Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "cloudflared" } }
}

if ($running) {
    Say "Tunnel already running (process $($running.Id))."
} else {
    if (Test-Path $TunnelLog) { Move-Item -Force $TunnelLog (Join-Path $Logs "tunnel.previous.log") }
    if (Test-Path $UrlFile) { Remove-Item -Force $UrlFile }
    Say "Starting the Cloudflare Quick Tunnel (a minimised cloudflared window will appear; leave it open)..."
    $cfArgs = @("tunnel", "--no-autoupdate", "--url", $Local, "--logfile", "`"$TunnelLog`"")
    $p = Start-Process -FilePath $Cloudflared -ArgumentList $cfArgs -WindowStyle Minimized -PassThru
    Set-Content -Path $PidFile -Value $p.Id
}

# 4. Public URL (a new address every time the tunnel starts)
$url = $null
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Path $TunnelLog) {
        $m = Select-String -Path $TunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($m) { $url = $m.Matches[-1].Value; break }
    }
    Start-Sleep -Seconds 1
}
if (-not $url) { throw "No public URL appeared in $TunnelLog within 60 seconds. Open that file to see the reason." }
Set-Content -Path $UrlFile -Value $url

Say "Checking the public address (DNS can take up to a minute)..."
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Url "$url/api/health" 10) { $ok = $true; break }
    Start-Sleep -Seconds 3
}
Write-Host ""
if ($ok) { Say "AIORI is public at:  $url" } else { Say "Tunnel started but the public check has not passed yet. Try in a minute:  $url" }
Say "The address is also saved in: $UrlFile"
