param([switch]$BuildEngines, [string]$Repository = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program" }
}

if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'This distribution requires 64-bit Windows and an NVIDIA GPU.'
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked 'py' @('-3.11', '-c', 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32')
        Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Invoke-Checked 'python' @('-c', 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32')
        Invoke-Checked 'python' @('-m', 'venv', '.venv')
    } else {
        throw 'Install 64-bit Python 3.11 first (https://www.python.org/downloads/windows/), then run this file again.'
    }
}
Invoke-Checked $python @('-c', 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32')

$fetchArgs = @('scripts\fetch_assets.py', '--groups', 'core', 'config')
if ($Repository) { $fetchArgs += @('--repo', $Repository) }
Invoke-Checked $python $fetchArgs
Invoke-Checked $python @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $python @('-m', 'pip', 'install', 'torch==2.11.0+cu128', 'torchvision==0.26.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'requirements.txt')
# InsightFace's onnxruntime requirement names the CPU distribution. Installing
# it with dependencies would put CPU and GPU packages in the same namespace.
# All its actual runtime dependencies are explicitly installed above.
Invoke-Checked $python @('-m', 'pip', 'install', '--no-deps', '.downloads\wheels\insightface-1.0.1-py3-none-any.whl')
if ($BuildEngines) {
    Invoke-Checked $python @('-m', 'pip', 'install', 'tensorrt==11.1.0.106', 'onnxslim==0.1.94')
    Invoke-Checked $python @('scripts\build_engines.py')
}
Invoke-Checked $python @('scripts\doctor.py', '--models')
Write-Host 'Installation completed. Configure camera credentials with .\.venv\Scripts\python.exe scripts\configure_cameras.py, then run .\start.ps1 and open http://localhost:8000'

