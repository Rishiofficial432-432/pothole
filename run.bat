@echo off
title Pothole Reporter
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
)
call "venv\Scripts\activate.bat"
python -c "import flask, ultralytics, cv2, PIL, geopy" >nul 2>nul
if errorlevel 1 (
    echo Installing packages...
    pip install flask ultralytics opencv-python pillow numpy geopy
)
rem Free port 5000 if an old server is still running
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"
python server.py
pause
