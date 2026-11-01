# Intent: one target-machine entrypoint for setup, verified data preparation and experiments.
# Status: parser/static verification locally; execution belongs to the school machine.
# Next: inspect bootstrap/suite diagnostics after target execution; no automatic fallback.
[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$WorkDir = "",
    [string]$RawDir = "",
    [string]$ReleaseDir = "",
    [string]$CacheDir = "",
    [switch]$PrepareOnly
)
$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
$environmentPython = Join-Path $projectRoot ".venv-train/Scripts/python.exe"

try {
    # Resolve custom relative paths against the caller's current folder before changing it.
    $WorkDir = if ($WorkDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkDir) } else { Join-Path $projectRoot "artifacts/school_run" }
    $RawDir = if ($RawDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RawDir) } else { Join-Path $projectRoot "data/raw/ptbxl/1.0.3" }
    $ReleaseDir = if ($ReleaseDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ReleaseDir) } else { Join-Path $projectRoot "data/processed/ptbxl_v1.0.3_500hz_minimal" }
    $CacheDir = if ($CacheDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CacheDir) } else { Join-Path $projectRoot "data/cache" }
    if ($PythonExe) { $PythonExe = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PythonExe) }
    Set-Location -LiteralPath $projectRoot

    $setupShellName = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh.exe" } else { "powershell.exe" }
    $setupShell = Join-Path $PSHOME $setupShellName
    if (-not (Test-Path -LiteralPath $setupShell -PathType Leaf)) { throw "PowerShell child executable not found: $setupShell" }
    $setupArguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "Setup-School.ps1"))
    if ($PythonExe) { $setupArguments += @("-PythonExe", $PythonExe) }
    Write-Host "School target only: setup first; data and model work starts only after setup succeeds."
    # Setup-School.ps1 calls exit; a child process prevents it terminating this wrapper.
    & $setupShell @setupArguments
    $setupExitCode = $LASTEXITCODE
    if ($setupExitCode -ne 0) {
        [Console]::Error.WriteLine("Setup failed (exit $setupExitCode). No data preparation or experiment command started.")
        exit $setupExitCode
    }
    if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) { throw "Setup returned success but .venv-train Python is missing." }

    $bootstrapCommand = if ($PrepareOnly) { "prepare" } else { "run" }
    $bootstrapArguments = @("-X", "utf8", (Join-Path $projectRoot "scripts/bootstrap_school.py"), $bootstrapCommand,
                            "--allow-download", "--raw-dir", $RawDir, "--release-dir", $ReleaseDir,
                            "--cache-dir", $CacheDir, "--work-dir", $WorkDir)
    Write-Host "Bootstrap command: $bootstrapCommand. Keep this terminal/session alive. Work directory: $WorkDir"
    & $environmentPython @bootstrapArguments
    $bootstrapExitCode = $LASTEXITCODE
    if ($bootstrapExitCode -ne 0) {
        [Console]::Error.WriteLine("Bootstrap incomplete (exit $bootstrapExitCode). Keep all folders; inspect console and $WorkDir/bootstrap_error.json or suite.log.")
    } elseif ($PrepareOnly) {
        Write-Host "Preparation finished. The verified local protocol is $WorkDir/target_protocol.json. No benchmark training was requested."
    } else {
        Write-Host "Suite completed. Return the newest results ZIP and checksum from $WorkDir/outbound."
    }
    exit $bootstrapExitCode
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
