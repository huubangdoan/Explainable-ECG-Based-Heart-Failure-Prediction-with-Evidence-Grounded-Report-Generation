[CmdletBinding()]
param(
    [string]$PythonExe = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
$environmentDir = Join-Path $projectRoot ".venv-train"
$environmentPython = Join-Path $environmentDir "Scripts/python.exe"
Set-Location -LiteralPath $projectRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    if ($PythonExe) {
        Invoke-Checked -Executable $PythonExe -Arguments @("-c", "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 required'")
        Invoke-Checked -Executable $PythonExe -Arguments @("-m", "venv", $environmentDir)
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked -Executable "py" -Arguments @("-3.12", "-c", "import sys; assert sys.version_info[:2] == (3,12)")
        Invoke-Checked -Executable "py" -Arguments @("-3.12", "-m", "venv", $environmentDir)
    } else {
        throw "Install 64-bit Python 3.12 from python.org, then rerun with -PythonExe 'C:\path\python.exe'. No system installer is run automatically."
    }
}
Invoke-Checked -Executable $environmentPython -Arguments @("-c", "import sys,struct; assert sys.version_info[:2] == (3,12); assert struct.calcsize('P') == 8, '64-bit Python required'")
Invoke-Checked -Executable $environmentPython -Arguments @("-m", "pip", "install", "--requirement", (Join-Path $projectRoot "requirements-ecg-train.txt"))
Invoke-Checked -Executable $environmentPython -Arguments @("-m", "pip", "install", "torch==2.14.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126")
Invoke-Checked -Executable $environmentPython -Arguments @("-m", "pip", "check")
Write-Host "Setup complete. No model has run. Start Run-School.ps1 on the school computer."
exit 0
