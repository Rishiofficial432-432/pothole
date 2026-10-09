<#
.SYNOPSIS
    Launcher for Streamlit Legacy Interface (port 8501)
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvStreamlit = Join-Path $ScriptDir "venv\Scripts\streamlit.exe"
if (-not (Test-Path $VenvStreamlit)) {
    Write-Error "Virtual environment not ready. Please run .\run.ps1 first."
    exit 1
}

# Free port 8501
$conn = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}

Write-Host "[STARTING] Launching Streamlit App at http://localhost:8501..." -ForegroundColor Green
& $VenvStreamlit run app.py
