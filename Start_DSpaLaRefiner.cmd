@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1" %*
if errorlevel 1 (
    echo.
    echo DSpaLaRefiner could not start. Read the error above.
    if "%~1"=="" pause
    exit /b 1
)


