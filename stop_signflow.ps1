<#
.SYNOPSIS
    Stops any running SignFlow backend and frontend processes.
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

Write-Host "Stopping SignFlow development processes..." -ForegroundColor Cyan

$Ports = @($BackendPort, $FrontendPort)
foreach ($p in $Ports) {
    try {
        $pids = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            if ($procId -gt 0) {
                Write-Host "  Terminating process on port $p (PID: $procId)..." -ForegroundColor Yellow
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}
}

Write-Host "All SignFlow processes stopped." -ForegroundColor Green

