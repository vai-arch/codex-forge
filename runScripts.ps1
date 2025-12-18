# --- Ensure codex environment is active before running this script ---

# Paths
$srcChunks = Join-Path $PSScriptRoot "..\dragon-codex\data\chunks"
$destChunks = Join-Path $PSScriptRoot "data\processed\chunks"

$srcIndexes = Join-Path $PSScriptRoot "..\dragon-codex\data\metadata\wiki"
$destIndexes = Join-Path $PSScriptRoot "data\corpus\indexes" 

# Ensure destination folder exists
if (-not (Test-Path $destChunks)) {
    New-Item -ItemType Directory -Path $destChunks -Force | Out-Null
}

Get-ChildItem -Path $srcChunks -Filter "*.jsonl" | Copy-Item -Destination $destChunks -Force
Get-ChildItem -Path $srcIndexes -Filter "*_index.json" | Copy-Item -Destination $destIndexes -Force

# List of Python modules to run
$modules = @(
    "src.training_pairs.01_generate_training_pairs"
)

foreach ($module in $modules) {
    Write-Host "Running Python module: $module"
    python -m $module

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Python module $module failed. Halting further execution."
        exit $LASTEXITCODE
    }
}

Write-Host "✅ All Python modules completed successfully."
