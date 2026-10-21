$ErrorActionPreference = 'Stop'
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/shutdown' -TimeoutSec 20

