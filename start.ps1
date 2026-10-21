$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run install.ps1 first.' }
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw 'Port 8000 is already in use. Check http://localhost:8000 before starting another instance.' }
& $python -u web\web_server.py
if ($LASTEXITCODE -ne 0) { throw "Detect exited with code $LASTEXITCODE" }

