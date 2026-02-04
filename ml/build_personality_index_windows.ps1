[
  CmdletBinding(SupportsShouldProcess)
]
param(
  [string]$ArchetypesDir = "..\archetypes",
  [string]$Meta = "..\archetypes\archetypes_meta.json",
  [string]$EmbeddingModel = "models\hand_embedding.keras",
  [string]$Out = "..\backend\palm_trait_index.npz",
  [int]$Img = 224,
  [int]$Batch = 64,
  [ValidateSet('mean','medoid')]
  [string]$Reduce = 'mean'
)

$ErrorActionPreference = "Stop"

$mlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $mlDir

Write-Output "[1/2] Ensuring ML venv + deps"
if ($PSCmdlet.ShouldProcess("ml/.venv", "Ensure venv + install requirements")) {
  if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
  }
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Output "[2/2] Building archetype index"
if ($PSCmdlet.ShouldProcess($Out, "Build archetype index")) {
  .\.venv\Scripts\python.exe build_trait_index.py --model $EmbeddingModel --data_dir $ArchetypesDir --meta $Meta --img $Img --batch $Batch --reduce $Reduce --out $Out
}

Write-Output "Done. Index written to: $Out"
