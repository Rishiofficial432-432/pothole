<#
.SYNOPSIS
    Unified Launcher for both Flask Web App (port 5000) and Streamlit App (port 8501)
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython    = Join-Path $ScriptDir "venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ScriptDir "venv\Scripts\streamlit.exe"

# Free port 5000 if occupied
$conn5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($conn5000) {
    Write-Host "[INFO] Freeing port 5000 (PID $($conn5000.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $conn5000.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# Free port 8501 if occupied
$conn8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($conn8501) {
    Write-Host "[INFO] Freeing port 8501 (PID $($conn8501.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $conn8501.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  Starting Flask Web Server on http://localhost:5000...  " -ForegroundColor Green
Write-Host "  Starting Streamlit App on http://localhost:8501...     " -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green

# Launch Streamlit in background process
Start-Process -FilePath $VenvStreamlit -ArgumentList "run app.py --server.headless false" -WorkingDirectory $ScriptDir

# Open Flask Web App in default browser
Start-Sleep -Seconds 2
Start-Process "http://localhost:5000"

# Run Flask server in foreground
& $VenvPython "$ScriptDir\server.py"
