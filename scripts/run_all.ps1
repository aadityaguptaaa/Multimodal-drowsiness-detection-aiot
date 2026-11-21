# Master Launcher: Starts both Backend and Frontend in separate PowerShell consoles
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING AIoT DRIVER DROWSINESS DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Root = Resolve-Path "$PSScriptRoot\.."

Write-Host "Starting Backend Service on http://127.0.0.1:5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; Write-Host 'AIoT Backend Server' -ForegroundColor Cyan; py -3.11 backend/app.py"

Start-Sleep -Seconds 2

Write-Host "Starting Frontend Server on http://localhost:5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; Write-Host 'AIoT Frontend Dashboard' -ForegroundColor Cyan; npm run dev"

Start-Sleep -Seconds 3

Write-Host "Opening Dashboard in default browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host "`nAll services launched successfully!" -ForegroundColor Green
