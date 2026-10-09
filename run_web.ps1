Set-Location $PSScriptRoot

# Free port 5000 if already in use
$conn = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($conn) {
    Write-Host "[INFO] Freeing port 5000 (terminating PID $($conn.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Start-Process "http://localhost:5000"
& "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\server.py"
