Param(
  [string]$ApiBase = "http://127.0.0.1:8000",
  [Parameter(Mandatory=$true)]
  [string]$ImagePath,
  [int]$TopK = 3
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ImagePath)) {
  throw "Image not found: $ImagePath"
}

Write-Host "Checking status..." -ForegroundColor Cyan
try {
  $status = Invoke-RestMethod -Method GET -Uri "$ApiBase/personality/status" -TimeoutSec 20
  $status | ConvertTo-Json -Depth 6
} catch {
  Write-Host "Could not call /personality/status. Is the backend running?" -ForegroundColor Yellow
}

Write-Host "\nCalling /scan-palm/personality..." -ForegroundColor Cyan

$form = @{
  file = Get-Item $ImagePath
  top_k = "$TopK"
}

$res = Invoke-RestMethod -Method POST -Uri "$ApiBase/scan-palm/personality" -Form $form -TimeoutSec 120
$res | ConvertTo-Json -Depth 8
