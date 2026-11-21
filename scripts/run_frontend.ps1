# PowerShell Script: Start AIoT Frontend Development Server
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  STARTING AIoT DROWSINESS DETECTION FRONTEND (React + Vite)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Dashboard URL: http://localhost:5173" -ForegroundColor Green

Set-Location -Path "$PSScriptRoot\..\frontend"
npm run dev
