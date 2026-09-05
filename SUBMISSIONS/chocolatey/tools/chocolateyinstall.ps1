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
    fileFullPath   = Join-Path $toolsDir 'reagent-0.2.0-py3-none-any.whl'
    url            = 'https://github.com/rugbedbugg/ReAgent/releases/download/v0.2.0/reagent-0.2.0-py3-none-any.whl'
    checksum       = 'a2746be9ec377bac6b720cb5653160db03ee7f52871d6f1b7baf26f92ed1659b'
    checksumType   = 'sha256'
}
Get-ChocolateyWebFile @wheelArgs

if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
& $python -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtualenv at $venvDir." }

$venvPython = Join-Path $venvDir 'Scripts\python.exe'

& $venvPython -m pip install --upgrade --quiet pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip inside the virtualenv." }

# Versions are pinned from the project's uv.lock. They are pinned without
# hashes deliberately: the lock is resolved per platform, and requiring hashes
# here would reject the Windows wheels that pip correctly selects.
& $venvPython -m pip install --quiet -r (Join-Path $toolsDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Failed to install ReAgent's dependencies." }

& $venvPython -m pip install --quiet --no-deps $wheelArgs.fileFullPath
if ($LASTEXITCODE -ne 0) { throw "Failed to install the ReAgent wheel." }

# Chocolatey shims every .exe it finds in the package directory. Only the two
# entry points should end up on PATH; without these markers the venv's
# python.exe and pip.exe would shadow the user's own.
Get-ChildItem -Path (Join-Path $venvDir 'Scripts') -Filter '*.exe' | ForEach-Object {
    if (@('reagent.exe', 'download_public_data.exe') -notcontains $_.Name) {
        New-Item -Path "$($_.FullName).ignore" -ItemType File -Force | Out-Null
    }
}

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
Write-Host "    reagent-download-data %LOCALAPPDATA%\reagent"
Write-Host "    reagent build-stock-cache"
Write-Host ""
Write-Host "The first pulls the pretrained USPTO policies and the ZINC stock, about 760 MB."
Write-Host "The second hashes that stock so later runs need roughly 0.6 GB of memory."
Write-Host ""
Write-Host "Then:  reagent plan `"CC(=O)Oc1ccccc1C(=O)O`""
