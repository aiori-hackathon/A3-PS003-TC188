@echo off
rem AIORI: double-click to start Docker, the AIORI site and the public Cloudflare tunnel.
title AIORI start
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-aiori.ps1" %*
echo.
echo You can close this window. The tunnel keeps running in its own minimised "cloudflared" window.
pause
