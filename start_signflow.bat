@echo off
setlocal
cd /d "%~dp0"

echo Launching SignFlow development supervisor via PowerShell...
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell was not found in PATH. Please install or enable Windows PowerShell.
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_signflow.ps1" %*
exit /b %errorlevel%

