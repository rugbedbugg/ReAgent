param([Parameter(Mandatory)][string]$InstallDir, [switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$bin = Join-Path $InstallDir 'bin'
$marker = Join-Path $InstallDir '.path-added'
function CheckExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}
$path = [Environment]::GetEnvironmentVariable('Path', 'User')
$entries = @($path -split ';' | Where-Object { $_ })
if ($Uninstall) {
    if (Test-Path $marker) {
        $entries = @($entries | Where-Object { $_.TrimEnd('\') -ine $bin.TrimEnd('\') })
        [Environment]::SetEnvironmentVariable('Path', ($entries -join ';'), 'User')
    }
    exit 0
}
$uv = Join-Path $InstallDir 'tools/uv.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'
$venv = Join-Path $InstallDir 'venv'
& $uv venv --clear --managed-python --python 3.11 $venv
CheckExit 'Create Python environment'
$python = Join-Path $venv 'Scripts/python.exe'
& $uv pip install --python $python -r (Join-Path $InstallDir 'requirements.txt')
CheckExit 'Install pinned dependencies'
$wheel = @(Get-ChildItem $InstallDir -Filter 'reagent-*-py3-none-any.whl')
if ($wheel.Count -ne 1) { throw 'Expected exactly one bundled ReAgent wheel' }
& $uv pip install --python $python --no-deps $wheel[0].FullName
CheckExit 'Install ReAgent'
if (-not ($entries | Where-Object { $_.TrimEnd('\') -ieq $bin.TrimEnd('\') })) {
    [Environment]::SetEnvironmentVariable('Path', (($entries + $bin) -join ';'), 'User')
    New-Item $marker -ItemType File -Force | Out-Null
}
