[CmdletBinding()]
param(
    [string]$WorkDir = "",
    [string]$ReleaseDir = "",
    [string]$RawDir = "",
    [string]$Config = "",
    [switch]$AllowDownload
)
$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
$environmentPython = Join-Path $projectRoot ".venv-train/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) { throw "Missing .venv-train. Run Setup-School.ps1 first." }
if (-not $WorkDir) { $WorkDir = Join-Path $projectRoot "artifacts/school_run" }
if (-not $ReleaseDir) { $ReleaseDir = Join-Path $projectRoot "data/processed/ptbxl_v1.0.3_500hz_minimal" }
if (-not $RawDir) { $RawDir = Join-Path $projectRoot "data/raw/ptbxl/1.0.3" }
if (-not $Config) { $Config = Join-Path $projectRoot "configs/ptbxl_experiments.json" }
$runArguments = @("-X", "utf8", (Join-Path $projectRoot "scripts/run_ecg_experiments.py"), "run-suite",
                  "--config", $Config, "--release-dir", $ReleaseDir, "--raw-dir", $RawDir,
                  "--work-dir", $WorkDir, "--device", "cuda")
if ($AllowDownload) { $runArguments += "--allow-download" }
Write-Host "Run/resume target suite. Keep this terminal/session alive. Results: $WorkDir"
& $environmentPython @runArguments
$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0) {
    Write-Host "Suite incomplete (exit $runExitCode). Keep work directory; read suite.log/error.json and return outbound diagnostics."
} else {
    Write-Host "Suite completed. Return the ZIP and checksum in $WorkDir/outbound."
}
exit $runExitCode
