Param(
  [ValidateSet('setup','run')]
  [string]$Mode = 'setup',
  [string]$VenvName = 'palm_env',
  [string]$HostIp = '0.0.0.0',
  [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BackendDir

$Py = "$VenvName\Scripts\python.exe"

function Initialize-Venv {
  if (-not (Test-Path $Py)) {
    Write-Host "Creating venv '$VenvName' with Python 3.11..."
    py -3.11 -m venv $VenvName
  }
}

function Install-Deps {
  Write-Host "Upgrading pip..."
  & $Py -m pip install --upgrade pip
  Write-Host "Installing backend requirements..."
  & $Py -m pip install -r requirements.txt
}

function Run-Api {
  Write-Host "Starting FastAPI on http://$HostIp`:$Port (reload enabled)..."
  & $Py -m uvicorn main:app --host $HostIp --port $Port --reload
}

Initialize-Venv

if ($Mode -eq 'setup') {
  Install-Deps
  Write-Host "Done. Run: .\setup_windows.ps1 -Mode run"
} else {
  Run-Api
}
