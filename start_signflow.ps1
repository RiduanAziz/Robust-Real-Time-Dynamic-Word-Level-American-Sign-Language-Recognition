<#
.SYNOPSIS
    One-command development supervisor for SignFlow (FastAPI backend + React frontend).
.DESCRIPTION
    Checks prerequisites, manages Python and Node environments, validates
    inference model assets, builds frontend production bundle, launches the
    FastAPI single-service web server, and opens the application in your browser.
.PARAMETER SkipInstall
    Skip dependency verification / installation checks for fast startup.
.PARAMETER NoBrowser
    Do not automatically open the browser once services are healthy.
.PARAMETER UiOnly
    Run in UI development mode: starts Vite live dev server (with hot reload) on FrontendPort.
.PARAMETER Public
    Create an instant 100% free public HTTPS domain via Cloudflare Quick Tunnel (no credit card or account needed).
    Allows opening and testing from smartphones (camera permissions enabled) and sharing with supervisors and examiners.
.PARAMETER Share
    Alias for -Public.
.PARAMETER Port
    Backend API and unified web application port (default: 8000).
.PARAMETER FrontendPort
    Vite dev server port when -UiOnly is active (default: 5173).
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$NoBrowser,
    [switch]$UiOnly,
    [switch]$Public,
    [switch]$Share,
    [int]$Port = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path "$PSScriptRoot").Path
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SignFlow - Robust Dynamic ASL Recognition Platform" -ForegroundColor Cyan
Write-Host "  One-Command Local Development Supervisor" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Prerequisite Checks
Write-Host "[1/6] Checking system prerequisites..." -ForegroundColor Yellow

$NodeCmd = Get-Command node -ErrorAction SilentlyContinue
$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $NodeCmd -or -not $NpmCmd) {
    Write-Error "Node.js and npm are required. Please install Node.js (v18+) from https://nodejs.org/"
    exit 1
}
$NodeVersion = & node -v
Write-Host "  + Node.js: $NodeVersion" -ForegroundColor Green

$PythonExe = $null
if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") {
    $PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
} else {
    $SysPython = Get-Command python -ErrorAction SilentlyContinue
    if ($SysPython) {
        $PythonExe = $SysPython.Source
    }
}

if (-not $PythonExe) {
    Write-Error "Python 3.10+ is required. Please install Python from https://www.python.org/downloads/"
    exit 1
}
$PyVersion = & $PythonExe --version 2>&1
Write-Host "  + Python: $PyVersion ($PythonExe)" -ForegroundColor Green

# 2. Virtual Environment Setup
Write-Host "[2/6] Verifying Python virtual environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creating virtual environment at .venv..." -ForegroundColor Cyan
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
}
$PythonExe = $VenvPython

# 3. Dependencies Verification
if (-not $SkipInstall) {
    Write-Host "[3/6] Verifying backend and frontend dependencies..." -ForegroundColor Yellow

    $BackendCheck = & $PythonExe -c "import fastapi, uvicorn, torch, cv2, mediapipe; print('OK')" 2>$null
    if ($BackendCheck -ne "OK") {
        Write-Host "  Installing backend runtime dependencies..." -ForegroundColor Cyan
        & $PythonExe -m pip install -e . --no-warn-script-location
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Editable install encountered warnings; falling back to requirements-runtime..."
            if (Test-Path "$RepoRoot\requirements-runtime.txt") {
                & $PythonExe -m pip install -r requirements-runtime.txt
            }
        }
    } else {
        Write-Host "  + Backend runtime packages verified." -ForegroundColor Green
    }

    $FrontendDir = Join-Path $RepoRoot "app\frontend"
    if (-not (Test-Path "$FrontendDir\node_modules")) {
        Write-Host "  Installing frontend dependencies with npm ci..." -ForegroundColor Cyan
        Push-Location $FrontendDir
        try {
            & npm.cmd ci
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "  + Frontend node_modules verified." -ForegroundColor Green
    }
} else {
    Write-Host "[3/6] Skipping dependency installation (-SkipInstall active)." -ForegroundColor DarkGray
}

# 4. Inference Asset Verification
Write-Host "[4/6] Verifying inference assets (checkpoints and MediaPipe)..." -ForegroundColor Yellow
$AssetCheckOutput = & $PythonExe "$RepoRoot\scripts\bootstrap_inference_assets.py" 2>&1
if ($LASTEXITCODE -ne 0) {
    if ($UiOnly) {
        Write-Warning "Inference assets are missing, but proceeding in UI-only mode (-UiOnly active)."
    } else {
        Write-Host "$AssetCheckOutput" -ForegroundColor Red
        Write-Error "Inference assets missing! Run 'python scripts/bootstrap_inference_assets.py' or pass -UiOnly for UI-only dev."
        exit 1
    }
} else {
    Write-Host "  + 3-Way Winning Ensemble and MediaPipe holistic task verified." -ForegroundColor Green
}

# Ensure production bundle exists if running unified single-container mode
$DistHtml = Join-Path $RepoRoot "app\frontend\dist\index.html"
if (-not $UiOnly -and -not (Test-Path $DistHtml)) {
    Write-Host "  Building frontend production bundle..." -ForegroundColor Cyan
    Push-Location (Join-Path $RepoRoot "app\frontend")
    try {
        & npm.cmd run build
    } finally {
        Pop-Location
    }
}

# 5. Free any stale processes on target ports
try {
    $StalePort = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($StalePort) {
        Write-Host "  Freeing stale process on port $Port (PID: $StalePort)..." -ForegroundColor DarkGray
        Stop-Process -Id $StalePort -Force -ErrorAction SilentlyContinue
    }
} catch {}

if ($UiOnly) {
    try {
        $StaleFPort = Get-NetTCPConnection -LocalPort $FrontendPort -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        if ($StaleFPort) {
            Write-Host "  Freeing stale process on port $FrontendPort (PID: $StaleFPort)..." -ForegroundColor DarkGray
            Stop-Process -Id $StaleFPort -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}

# 6. Launch Services
Write-Host "[5/6] Starting SignFlow services..." -ForegroundColor Yellow

$env:PYTHONPATH = "$RepoRoot\src"
$env:PORT = "$Port"

$BackendArgs = @("-m", "uvicorn", "sign_language.api.main:app", "--host", "127.0.0.1", "--port", "$Port")
$BackendProc = Start-Process -FilePath $PythonExe -ArgumentList $BackendArgs -WorkingDirectory $RepoRoot -PassThru
Write-Host "  + Backend process PID: $($BackendProc.Id) on port $Port" -ForegroundColor Green

$FrontendProc = $null
if ($UiOnly) {
    Write-Host "  Starting Vite frontend dev server on port $FrontendPort..." -ForegroundColor Cyan
    $FrontendProc = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--port", "$FrontendPort", "--host", "127.0.0.1" -WorkingDirectory "$RepoRoot\app\frontend" -PassThru
    Write-Host "  + Frontend dev process PID: $($FrontendProc.Id)" -ForegroundColor Green
}

# 7. Health Check Polling
Write-Host "[6/6] Waiting for service health checks..." -ForegroundColor Yellow
$HealthUrl = "http://127.0.0.1:$Port/health/live"
$MaxAttempts = 40
$Healthy = $false

for ($i = 1; $i -le $MaxAttempts; $i++) {
    Start-Sleep -Seconds 1
    if ($BackendProc.HasExited) {
        Write-Error "Backend process exited prematurely with exit code $($BackendProc.ExitCode)!"
        exit 1
    }
    try {
        $Resp = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2 -ErrorAction Stop
        if ($Resp.status -eq "ok" -or $Resp.status -eq "alive" -or $Resp.process_alive -eq $true) {
            $Healthy = $true
            break
        }
    } catch {}
    Write-Host "  Waiting for backend startup ($i/$MaxAttempts)..." -ForegroundColor DarkGray
}

if (-not $Healthy) {
    Write-Error "Backend health check failed after $MaxAttempts seconds."
    exit 1
}

# 8. Cloudflare Quick Tunnel (Free Public HTTPS & WSS for Mobile & Examiners)
$PublicUrl = $null
$CloudflaredProc = $null
$CloudflaredLog = $null

if ($Public -or $Share) {
    Write-Host "`n[Public Tunnel] Establishing Cloudflare Quick Tunnel (100% Free HTTPS)..." -ForegroundColor Yellow
    $CloudflaredCmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $CloudflaredCmd) {
        Write-Warning "cloudflared is not installed or not in PATH. Falling back to local URL."
    } else {
        $CloudflaredLog = [System.IO.Path]::GetTempFileName()
        $TargetTunnelPort = if ($UiOnly) { $FrontendPort } else { $Port }
        $CloudflaredProc = Start-Process -FilePath $CloudflaredCmd.Source -ArgumentList "tunnel", "--url", "http://127.0.0.1:$TargetTunnelPort", "--no-autoupdate" -RedirectStandardError $CloudflaredLog -PassThru -WindowStyle Hidden

        Write-Host "  Connecting to Cloudflare edge..." -ForegroundColor DarkGray
        for ($w = 1; $w -le 25; $w++) {
            Start-Sleep -Seconds 1
            if (Test-Path $CloudflaredLog) {
                $LogTxt = Get-Content $CloudflaredLog -Raw -ErrorAction SilentlyContinue
                if ($LogTxt -match '\|\s*(https://[a-zA-Z0-9-]+\.trycloudflare\.com)\s*\|') {
                    $PublicUrl = $matches[1]
                    break
                }
            }
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  SignFlow is LIVE and healthy!" -ForegroundColor Green
if ($PublicUrl) {
    Write-Host "  ==========================================================" -ForegroundColor Magenta
    Write-Host "  PUBLIC LIVE HTTPS DOMAIN (FOR SUPERVISOR & EXAMINERS):" -ForegroundColor Magenta
    Write-Host "  $PublicUrl" -ForegroundColor Yellow
    Write-Host "  ==========================================================" -ForegroundColor Magenta
    Write-Host "  + 100% Free - No credit card, no account required!" -ForegroundColor Green
    Write-Host "  + Mobile Camera: Fully works (HTTPS grants camera permission)" -ForegroundColor DarkGray
    Write-Host "  + Secure WebSocket (wss://): Auto-connected by SignFlow" -ForegroundColor DarkGray
    try {
        Set-Clipboard -Value $PublicUrl
        Write-Host "  + Link copied to clipboard automatically!" -ForegroundColor Green
    } catch {}
} elseif ($UiOnly) {
    Write-Host "  - Frontend UI (Vite Dev): http://127.0.0.1:$FrontendPort" -ForegroundColor Cyan
    Write-Host "  - Backend API:            http://127.0.0.1:$Port" -ForegroundColor Cyan
} else {
    Write-Host "  - Application URL:        http://127.0.0.1:$Port/" -ForegroundColor Cyan
    Write-Host "  - Liveness Check:         http://127.0.0.1:$Port/health/live" -ForegroundColor Cyan
    Write-Host "  - Readiness Check:        http://127.0.0.1:$Port/health/ready" -ForegroundColor Cyan
    Write-Host "  - Model Info:             http://127.0.0.1:$Port/model/info" -ForegroundColor Cyan
}
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Green

# Open browser
if (-not $NoBrowser) {
    $TargetUrl = if ($PublicUrl) { $PublicUrl } elseif ($UiOnly) { "http://127.0.0.1:$FrontendPort" } else { "http://127.0.0.1:$Port" }
    Start-Process $TargetUrl
}

# Keep supervisor running until interrupted
try {
    while ($true) {
        if ($BackendProc.HasExited) {
            Write-Warning "Backend process terminated unexpectedly (ExitCode: $($BackendProc.ExitCode))."
            break
        }
        if ($UiOnly -and $FrontendProc -and $FrontendProc.HasExited) {
            Write-Warning "Frontend process terminated unexpectedly."
            break
        }
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "`nShutting down SignFlow services..." -ForegroundColor Cyan
    if ($CloudflaredProc -and -not $CloudflaredProc.HasExited) {
        Stop-Process -Id $CloudflaredProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($CloudflaredLog -and (Test-Path $CloudflaredLog)) {
        Remove-Item $CloudflaredLog -Force -ErrorAction SilentlyContinue
    }
    if ($BackendProc -and -not $BackendProc.HasExited) {
        Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($FrontendProc -and -not $FrontendProc.HasExited) {
        Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
    }
    try {
        $BPort = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        if ($BPort) { Stop-Process -Id $BPort -Force -ErrorAction SilentlyContinue }
    } catch {}
    if ($UiOnly) {
        try {
            $FPort = Get-NetTCPConnection -LocalPort $FrontendPort -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
            if ($FPort) { Stop-Process -Id $FPort -Force -ErrorAction SilentlyContinue }
        } catch {}
    }
    Write-Host "SignFlow stopped cleanly." -ForegroundColor Green
}
