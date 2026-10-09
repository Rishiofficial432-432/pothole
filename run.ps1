<#
.SYNOPSIS
    One-Command Launcher for CivicRoad HTML / CSS / JS Web Application (port 5000)
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "       CIVIC ROAD SURFACE INTEGRITY & POTHOLE AI         " -ForegroundColor Cyan
Write-Host "         Modern HTML / CSS / JS Web Dashboard            " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
$VenvPip    = Join-Path $ScriptDir "venv\Scripts\pip.exe"
$VenvAct    = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"

# 1. Check or Create Venv
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/5] Virtual environment not found. Locating Python installation..." -ForegroundColor Yellow
    
    $pythonCmd = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    } elseif (Test-Path "C:\Program Files\Python313\python.exe") {
        $pythonCmd = "C:\Program Files\Python313\python.exe"
    } elseif (Test-Path "C:\Program Files\Python312\python.exe") {
        $pythonCmd = "C:\Program Files\Python312\python.exe"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
        $pythonCmd = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") {
        $pythonCmd = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe") {
        $pythonCmd = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    } else {
        Write-Error "Python 3.10+ was not found. Please install Python or add it to PATH."
        exit 1
    }

    Write-Host "[INFO] Creating venv using $pythonCmd..." -ForegroundColor Yellow
    & $pythonCmd -m venv venv
    Write-Host "[OK] Venv created." -ForegroundColor Green
} else {
    Write-Host "[1/5] Virtual environment 'venv' found." -ForegroundColor Green
}

# 2. Activate venv
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path $VenvAct) {
    try { & $VenvAct } catch { $env:VIRTUAL_ENV = Join-Path $ScriptDir "venv"; $env:PATH = "$(Join-Path $ScriptDir 'venv\Scripts');$env:PATH" }
} else {
    $env:VIRTUAL_ENV = Join-Path $ScriptDir "venv"
    $env:PATH = "$(Join-Path $ScriptDir 'venv\Scripts');$env:PATH"
}

# 3. Verify packages
Write-Host "[3/5] Verifying dependencies from requirements.txt..." -ForegroundColor Yellow
$pkgCheck = & $VenvPython -c "import flask, ultralytics, cv2, PIL, imageio_ffmpeg; print('OK')" 2>$null
if ($pkgCheck -ne "OK") {
    Write-Host "[INFO] Installing / updating required dependencies..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPip install -r requirements.txt
    Write-Host "[OK] Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "[OK] All required dependencies ready." -ForegroundColor Green
}

# 4. Check Database
Write-Host "[4/5] Checking potholes.db database..." -ForegroundColor Yellow
$DbPath = Join-Path $ScriptDir "potholes.db"
if (-not (Test-Path $DbPath)) {
    Write-Host "[INFO] Seeding initial database with 200+ Indian road locations..." -ForegroundColor Yellow
    & $VenvPython "scrape_india_data.py" --source seed
    Write-Host "[OK] Database seeded." -ForegroundColor Green
} else {
    Write-Host "[OK] Database verified." -ForegroundColor Green
}

# 5. Free port 5000 if occupied
$conn = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($conn) {
    Write-Host "[INFO] Freeing port 5000 (terminating PID $($conn.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 6. Launch Browser and Server
Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  [LAUNCHING] Starting HTML / CSS / JS Web Dashboard...  " -ForegroundColor Green
Write-Host "  Local Dashboard: http://localhost:5000                  " -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""

Start-Process "http://localhost:5000"

& $VenvPython "server.py"
