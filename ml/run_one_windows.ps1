<#$
.SYNOPSIS
  Guided Windows runner for one-image inference (infer_one.py)

.DESCRIPTION
  Creates/uses ml/.venv (Python 3.11), optionally installs requirements, prompts
  for model + image paths, and auto-detects model kind (predict vs embed).
#>

[CmdletBinding()]
param(
  [switch]$SkipInstall,

  [ValidateSet('predict', 'embed')]
  [string]$Mode = ''
)

$ErrorActionPreference = 'Stop'

$mlDir = $PSScriptRoot
Set-Location -LiteralPath $mlDir

$PythonExe = Join-Path $mlDir '.venv\Scripts\python.exe'

function Initialize-MLVenv {
  [CmdletBinding(SupportsShouldProcess)]
  param(
    [switch]$SkipInstall
  )

  if (-not (Test-Path -LiteralPath $PythonExe)) {
    if ($PSCmdlet.ShouldProcess('.venv', 'Create Python 3.11 virtual environment')) {
      Write-Output 'Creating ml/.venv (Python 3.11)'
      py -3.11 -m venv .venv
    }
  }

  if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Expected venv python at '$PythonExe' but it was not found. Is Python 3.11 installed (py -3.11)?"
  }

  if (-not $SkipInstall) {
    if ($PSCmdlet.ShouldProcess('requirements.txt', 'Install ML requirements')) {
      Write-Output 'Installing ML requirements'
      & $PythonExe -m pip install --upgrade pip
      & $PythonExe -m pip install -r requirements.txt
    }
  } else {
    Write-Verbose 'SkipInstall set; not installing requirements.'
  }
}

function Get-Choice($prompt, $choices) {
  Write-Output ''
  Write-Output $prompt
  for ($i = 0; $i -lt $choices.Count; $i += 1) {
    Write-Output ("[{0}] {1}" -f ($i + 1), $choices[$i])
  }
  while ($true) {
    $raw = Read-Host "Enter choice (1-$($choices.Count))"
    if ($raw -match "^\d+$") {
      $n = [int]$raw
      if ($n -ge 1 -and $n -le $choices.Count) { return $choices[$n - 1] }
    }
    Write-Warning 'Invalid choice.'
  }
}

function Get-ExistingPath($prompt, $default) {
  while ($true) {
    $raw = Read-Host "$prompt`n(default: $default)"
    $p = if ([string]::IsNullOrWhiteSpace($raw)) { $default } else { $raw }

    # Accept both relative and absolute.
    $full = $p
    if (-not [System.IO.Path]::IsPathRooted($p)) {
      $full = Join-Path -Path (Get-Location) -ChildPath $p
    }

    if (Test-Path -LiteralPath $full) {
      return $p
    }

    Write-Warning "Not found: $full"
  }
}

Initialize-MLVenv -SkipInstall:$SkipInstall

$modelPath = Get-ExistingPath "Model path" "models\hand_cnn.keras"

$cmdMode = $Mode
if ([string]::IsNullOrWhiteSpace($cmdMode)) {
  try {
    $json = & $PythonExe detect_model_kind.py --model $modelPath
    $info = $json | ConvertFrom-Json
    if ($null -ne $info -and $null -ne $info.mode) {
      $cmdMode = [string]$info.mode
      if ($null -ne $info.out_dim -and $info.out_dim -ne "") {
        Write-Verbose ("Auto-detected model kind: out_dim={0}, mode={1}" -f $info.out_dim, $cmdMode)
        Write-Output ("Model: {0} | out_dim={1} | mode={2}" -f $modelPath, $info.out_dim, $cmdMode)
      } else {
        Write-Verbose ("Auto-detected model kind: mode={0}" -f $cmdMode)
        Write-Output ("Model: {0} | mode={1}" -f $modelPath, $cmdMode)
      }
    }
  } catch {
    Write-Warning ("Auto-detection failed; falling back to manual mode selection. {0}" -f $_.Exception.Message)
  }
}

if ([string]::IsNullOrWhiteSpace($cmdMode)) {
  $choice = Get-Choice "Could not auto-detect mode. Choose:" @(
    "predict (class probabilities)",
    "embed (feature vector)"
  )
  $cmdMode = if ($choice.StartsWith("predict")) { "predict" } else { "embed" }
}

# Prefer sample_images as a safe drop zone.
$defaultImage = "sample_images\palm1.jpg"
$imagePath = Get-ExistingPath "Image path" $defaultImage

$imgSizeRaw = Read-Host "Resize (img). Press Enter for 224"
$imgSize = if ([string]::IsNullOrWhiteSpace($imgSizeRaw)) { 224 } else { [int]$imgSizeRaw }

Write-Output ''
Write-Verbose 'Note: first run may take a moment while TensorFlow initializes.'
Write-Output 'Running infer_one.py...'

& $PythonExe infer_one.py --mode $cmdMode --model $modelPath --image $imagePath --img $imgSize
