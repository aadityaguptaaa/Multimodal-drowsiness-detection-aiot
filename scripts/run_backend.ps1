# PowerShell Script: Start AIoT Backend Server
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  STARTING AIoT DROWSINESS DETECTION BACKEND (Flask + Socket.IO)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "API Server URL: http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Health Check:  http://127.0.0.1:5000/api/health" -ForegroundColor Yellow

py -3.11 backend/app.py
