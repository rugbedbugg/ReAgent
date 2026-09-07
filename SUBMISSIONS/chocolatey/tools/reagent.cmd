@echo off
setlocal
if not defined REAGENT_DATA set "REAGENT_DATA=%LOCALAPPDATA%\reagent"
"%~dp0venv\Scripts\reagent.exe" %*
exit /b %ERRORLEVEL%
