[
  CmdletBinding(SupportsShouldProcess)
]
param(
  [string]$RawSource = "C:\Users\dasne\Downloads\archive",
  [string]$SplitTarget = "C:\Users\dasne\Downloads\palm_gender_dataset",
  [string]$Classes = "male,female",
  [int]$Epochs = 10,
  [int]$Img = 224,
  [int]$Batch = 32,
  [switch]$SkipSplit
)

$ErrorActionPreference = "Stop"

$mlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $mlDir

Write-Output "[1/4] Ensuring venv + deps"
if ($PSCmdlet.ShouldProcess("ml/.venv", "Ensure venv + install requirements")) {
  if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
  }
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

if (-not $SkipSplit) {
  Write-Output "[2/4] Splitting dataset"
  if ($PSCmdlet.ShouldProcess($SplitTarget, "Split dataset")) {
    .\.venv\Scripts\python.exe split_dataset.py --source $RawSource --target $SplitTarget --classes $Classes --train 0.7 --val 0.15 --test 0.15
  }
}

Write-Output "[3/4] Dataset report"
if ($PSCmdlet.ShouldProcess($SplitTarget, "Generate dataset report")) {
  .\.venv\Scripts\python.exe dataset_report.py --root $SplitTarget --classes $Classes
}

Write-Output "[4/4] Training + export"
if ($PSCmdlet.ShouldProcess($mlDir, "Train model + export TFLite")) {
  .\.venv\Scripts\python.exe train_cnn.py --mode directory --data $SplitTarget --epochs $Epochs --img $Img --batch $Batch
  .\.venv\Scripts\python.exe export_tflite.py --keras_model models\hand_embedding.keras --out models\hand_embedding_float16.tflite --quant float16
}

Write-Output "Done. Models are in: $mlDir\models"
