@echo off
title CivicRoad - Launch Both Servers
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
pause
