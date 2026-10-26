$ErrorActionPreference = 'Stop'

$toolsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvDir  = Join-Path $toolsDir 'venv'

# ReAgent is a Python package, not a standalone binary. RDKit, ONNX Runtime and
# AiZynthFinder all cap out below Python 3.12, so it gets its own 3.11 venv
# rather than being installed into whatever interpreter happens to be on PATH.
# The python311 dependency in the nuspec guarantees one is present.
$python = $null
foreach ($candidate in @(
    "$env:SystemDrive\Python311\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "${env:ProgramFiles(x86)}\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)) {
    if (Test-Path $candidate) { $python = $candidate; break }
}

if (-not $python) {
    # Fall back to the launcher, which the python311 package also registers.
    $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($launcher) {
        $resolved = & $launcher.Source -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) { $python = $resolved.Trim() }
    }
}

if (-not $python) {
    throw "Could not find a Python 3.11 interpreter. The python311 package should have provided one; try 'choco install python311' and reinstall."
}

Write-Host "Using interpreter: $python"

# The ReAgent wheel comes from the GitHub release rather than PyPI: the name
# 'reagent' on PyPI belongs to an unrelated project (Facebook's reinforcement
# learning library), so 'pip install reagent' would fetch the wrong package.
$wheelArgs = @{
    packageName    = 'reagent'
    fileFullPath   = Join-Path $toolsDir 'reagent-0.3.0-py3-none-any.whl'
    url            = 'https://github.com/rugbedbugg/ReAgent/releases/download/v0.3.0/reagent-0.3.0-py3-none-any.whl'
    checksum       = 'e0fa63990c1599b8ee900f66289397653a6dc43fe723250d3a5c4002170d83b2'
    checksumType   = 'sha256'
}
Get-ChocolateyWebFile @wheelArgs

# The verifier runs without an interactive console. Use Chocolatey's process
# helper to capture output and reject nonzero exits, and give Python EOF on
# stdin so neither venv/ensurepip nor pip can wait for input from that host.
function Invoke-ReAgentPython {
    param([string]$Executable, [string]$Arguments, [string]$Stage)

    Write-Host "ReAgent: $Stage"
    Start-ChocolateyProcessAsAdmin -ExeToRun $env:ComSpec `
        -Statements "/d /s /c `"`"$Executable`" -I -u $Arguments <NUL`"" `
        -ValidExitCodes @(0) -NoSleep
    Write-Host "ReAgent: $Stage completed"
}

if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
Invoke-ReAgentPython -Executable $python -Stage 'Create virtualenv' `
    -Arguments "-m venv --without-pip `"$venvDir`""

$venvPython = Join-Path $venvDir 'Scripts\python.exe'

# Keep pip bootstrap separate so a failed verifier log names the stalled stage.
Invoke-ReAgentPython -Executable $venvPython -Stage 'Bootstrap pip' `
    -Arguments '-m ensurepip --upgrade --default-pip'

$pipInstall = '-m pip --disable-pip-version-check --no-input install --progress-bar off --only-binary=:all:'
Invoke-ReAgentPython -Executable $venvPython -Stage 'Upgrade pip' `
    -Arguments "$pipInstall --upgrade pip"

# Versions are pinned from the project's uv.lock. They are pinned without
# hashes deliberately: the lock is resolved per platform, and requiring hashes
# here would reject the Windows wheels that pip correctly selects.
$requirements = Join-Path $toolsDir 'requirements.txt'
Invoke-ReAgentPython -Executable $venvPython -Stage 'Install dependencies' `
    -Arguments "$pipInstall -r `"$requirements`""

Invoke-ReAgentPython -Executable $venvPython -Stage 'Install ReAgent wheel' `
    -Arguments "$pipInstall --no-deps `"$($wheelArgs.fileFullPath)`""

Invoke-ReAgentPython -Executable $venvPython -Stage 'Check installed dependencies' `
    -Arguments '-m pip --disable-pip-version-check --no-input check'

# All entry points are registered explicitly below. The reagent wrapper sets
# the per-user data path at runtime, rather than capturing the installer's home.
Get-ChildItem -Path (Join-Path $venvDir 'Scripts') -Filter '*.exe' | ForEach-Object {
    New-Item -Path "$($_.FullName).ignore" -ItemType File -Force | Out-Null
}
Install-BinFile -Name 'reagent' -Path (Join-Path $toolsDir 'reagent.cmd')

# AiZynthFinder names its fetcher download_public_data, which is too generic to
# put on a shared PATH. Shim it under the project's own prefix instead.
$downloader = Join-Path $venvDir 'Scripts\download_public_data.exe'
if (Test-Path $downloader) {
    New-Item -Path "$downloader.ignore" -ItemType File -Force | Out-Null
    Install-BinFile -Name 'reagent-download-data' -Path $downloader
}

Write-Host ""
Write-Host "ReAgent is installed. Two one-time steps before the first plan:"
Write-Host ""
Write-Host 'In Command Prompt:'
Write-Host '    reagent-download-data "%LOCALAPPDATA%\reagent"'
Write-Host "    reagent build-stock-cache"
Write-Host ""
Write-Host "The first pulls the pretrained USPTO policies and the ZINC stock, about 760 MB."
Write-Host "The second hashes that stock so later runs need roughly 0.6 GB of memory."
Write-Host ""
Write-Host "Then:  reagent plan `"CC(=O)Oc1ccccc1C(=O)O`""
