$ErrorActionPreference = 'Stop'

$toolsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvDir  = Join-Path $toolsDir 'venv'

# Install-BinFile was called explicitly for this one, so it has to be removed
# explicitly. The reagent.exe shim is auto-generated and auto-removed.
$downloader = Join-Path $venvDir 'Scripts\download_public_data.exe'
Uninstall-BinFile -Name 'reagent-download-data' -Path $downloader

if (Test-Path $venvDir) {
    Remove-Item -Recurse -Force $venvDir
}

# Model data and stock caches live outside the package directory and are the
# user's, not ours. Say where they are rather than deleting them.
Write-Host ""
Write-Host "Removed the ReAgent virtualenv."
Write-Host "Downloaded model data and stock caches were left in place."
Write-Host "Delete them by hand if you want the disk space back, typically:"
Write-Host "    %LOCALAPPDATA%\reagent"
