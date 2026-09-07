param([Parameter(Mandatory)][ValidateSet('chocolatey','winget')][string]$Channel)
$ErrorActionPreference = 'Stop'
function CheckExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}
if ($Channel -eq 'chocolatey') {
    Push-Location prepared/chocolatey
    choco pack
    CheckExit 'Pack'
    try {
        choco install reagent --source '.;https://community.chocolatey.org/api/v2/' --yes --no-progress
        CheckExit 'Install'
        reagent --help
        CheckExit 'CLI smoke'
        reagent-download-data --help
        CheckExit 'Downloader smoke'
    } finally {
        choco uninstall reagent --yes --no-progress
        CheckExit 'Uninstall'
        Pop-Location
    }
    exit 0
}
$version = (Get-Content prepared/release.json -Raw | ConvertFrom-Json).version
$packageDir = (Resolve-Path prepared/winget).Path
$installer = "$packageDir/reagent-$version-windows-x86_64-setup.exe"
if (-not (Test-Path $installer)) {
$uv = (mise which uv).Trim()
CheckExit 'Locate uv'
Copy-Item $uv "$packageDir/payload/uv.exe"
$uvVersion = ((& $uv --version) -split ' ')[1]
Invoke-WebRequest "https://raw.githubusercontent.com/astral-sh/uv/$uvVersion/LICENSE-MIT" -OutFile "$packageDir/payload/uv-LICENSE-MIT.txt"
$compiler = "${env:ProgramFiles(x86)}/Inno Setup 6/ISCC.exe"
if (-not (Test-Path $compiler)) { throw 'Inno Setup is missing from the Windows build image' }
& $compiler "/DAppVersion=$version" "/DPayloadDir=$packageDir/payload" "/O$packageDir" SUBMISSIONS/winget/reagent.iss
CheckExit 'Build installer'
}
mise exec -- python scripts/packaging/winget.py --version $version --installer $installer --output "$packageDir/manifests"
CheckExit 'Generate release manifests'
Install-Module Microsoft.WinGet.Client -Repository PSGallery -Force
Repair-WinGetPackageManager -AllUsers
winget settings --enable LocalManifestFiles
CheckExit 'Enable local manifests'
winget validate --manifest "$packageDir/manifests"
CheckExit 'Validate release manifests'
# The installer is an unpublished CI artifact. Test the exact same bytes using
# a loopback URL; publication later verifies the final release URL before a PR.
$localManifests = Join-Path $env:RUNNER_TEMP 'reagent-local-manifests'
mise exec -- python scripts/packaging/winget.py --version $version --installer $installer --output $localManifests --url "http://127.0.0.1:8765/reagent-$version-windows-x86_64-setup.exe"
CheckExit 'Generate local test manifests'
$python = (mise which python).Trim()
$server = Start-Process $python -ArgumentList @('-m','http.server','8765','--bind','127.0.0.1','--directory',"`"$packageDir`"") -PassThru
$installed = $false
$root = Join-Path $env:LOCALAPPDATA 'Programs/ReAgent'
$originalPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$data = Join-Path $env:LOCALAPPDATA 'reagent/packaging-test-marker'
New-Item (Split-Path $data) -ItemType Directory -Force | Out-Null
New-Item $data -ItemType File -Force | Out-Null
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { Invoke-WebRequest "http://127.0.0.1:8765/" -Method Head | Out-Null; $ready = $true; break }
        catch { Start-Sleep 1 }
    }
    if (-not $ready) { throw 'Local installer server did not start' }
    winget install --manifest $localManifests --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
    CheckExit 'Install'
    $installed = $true
    & "$root/bin/reagent.cmd" --help
    CheckExit 'CLI smoke'
    & "$root/bin/reagent-download-data.cmd" --help
    CheckExit 'Downloader smoke'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not ($userPath -split ';' | Where-Object { $_ -ieq (Join-Path $root 'bin') })) { throw 'CLI path was not registered' }
    Remove-Item Env:REAGENT_DATA -ErrorAction SilentlyContinue
    $ErrorActionPreference = 'Continue'
    $output = & "$root/bin/reagent.cmd" build-stock-cache 2>&1 | Out-String
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0 -or -not $output.Contains((Join-Path $env:LOCALAPPDATA 'reagent'))) { throw 'Default data directory is incorrect' }
} finally {
    if ($installed) {
        winget uninstall --manifest $localManifests --scope user --silent --accept-source-agreements --disable-interactivity
        CheckExit 'Uninstall'
        if (Test-Path "$root/bin/reagent.cmd") { throw 'Launcher survived uninstall' }
        if (-not (Test-Path $data)) { throw 'Uninstall removed user data' }
        $after = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (([string]$after).TrimEnd(';') -ne ([string]$originalPath).TrimEnd(';')) { throw 'Uninstall did not preserve the original user PATH' }
    }
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
