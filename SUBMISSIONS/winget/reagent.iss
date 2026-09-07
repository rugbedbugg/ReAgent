#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef PayloadDir
  #error PayloadDir is required
#endif
[Setup]
AppId=ReAgent
AppName=ReAgent
AppVersion={#AppVersion}
AppPublisher=Partha Pratim Gogoi
AppPublisherURL=https://github.com/rugbedbugg/ReAgent
DefaultDirName={localappdata}\Programs\ReAgent
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=reagent-{#AppVersion}-windows-x86_64-setup
Compression=lzma2
SolidCompression=yes
ChangesEnvironment=yes
UninstallDisplayName=ReAgent
CloseApplications=no

[Files]
Source: "{#PayloadDir}\uv-LICENSE-MIT.txt"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#PayloadDir}\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#PayloadDir}\reagent-{#AppVersion}-py3-none-any.whl"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "reagent.cmd"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "reagent-download-data.cmd"; DestDir: "{app}\bin"; Flags: ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\python"
Type: files; Name: "{app}\.path-added"

[Code]
procedure ConfigureApp(Uninstall: Boolean);
var
  Args: String;
  ResultCode: Integer;
begin
  Args := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\setup.ps1') + '" -InstallDir "' + ExpandConstant('{app}') + '"';
  if Uninstall then Args := Args + ' -Uninstall';
  if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Args, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    RaiseException('Could not start ReAgent setup.');
  if ResultCode <> 0 then
    RaiseException('ReAgent setup failed. Check your internet connection and retry.');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    WizardForm.StatusLabel.Caption := 'Setting up ReAgent...';
    ConfigureApp(False);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then ConfigureApp(True);
end;
