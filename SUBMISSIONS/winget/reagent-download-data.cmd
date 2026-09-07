@echo off
"%~dp0..\venv\Scripts\download_public_data.exe" %*
exit /b %ERRORLEVEL%
