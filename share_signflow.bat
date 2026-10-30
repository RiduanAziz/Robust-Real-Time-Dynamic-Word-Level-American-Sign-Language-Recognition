@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   SignFlow — Launching Free Public HTTPS Domain
echo   (No credit card, no account required)
echo ============================================================
echo.

where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell was not found in PATH.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_signflow.ps1" -Public %*
if %errorlevel% neq 0 (
    echo.
    echo [SignFlow stopped with an error or exit code %errorlevel%]
    pause
)

