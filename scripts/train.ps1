# PowerShell Script: AIoT Model Training Pipeline
param(
    [switch]$SmokeTest = $false,
    [int]$Samples = 600,
    [int]$Epochs = 15,
    [int]$BatchSize = 32
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AIoT DRIVER DROWSINESS DETECTION - MODEL TRAINING" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Inspect Dataset
Write-Host "`n[Step 1/3] Inspecting Driver Drowsiness Dataset (DDD)..." -ForegroundColor Yellow
py -3.11 backend/training/inspect_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dataset inspection failed. Aborting." -ForegroundColor Red
    exit 1
}

# Step 2: Train MobileNetV2
Write-Host "`n[Step 2/3] Training MobileNetV2 Transfer Learning Model..." -ForegroundColor Yellow
if ($SmokeTest) {
    Write-Host "Running in SMOKE-TEST verification mode (samples: $Samples)..." -ForegroundColor Magenta
    py -3.11 backend/training/train_model.py --smoke-test --samples $Samples --epochs 2 --batch-size 16
} else {
    Write-Host "Running in FULL DATASET training mode (Epochs: $Epochs, Batch Size: $BatchSize)..." -ForegroundColor Green
    py -3.11 backend/training/train_model.py --epochs $Epochs --batch-size $BatchSize
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Model training failed. Aborting." -ForegroundColor Red
    exit 1
}

# Step 3: Evaluate CNN on Held-Out Test Set
Write-Host "`n[Step 3/4] Evaluating CNN Model on Held-Out Test Set..." -ForegroundColor Yellow
py -3.11 backend/training/evaluate_model.py

# Step 4: Train PERCLOS Machine Learning Model
Write-Host "`n[Step 4/4] Training PERCLOS Machine Learning Classifier (Paper #8: Kumar et al. 2023)..." -ForegroundColor Yellow
py -3.11 backend/training/train_perclos_model.py

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  TRAINING & EVALUATION COMPLETE! Artifacts saved to backend/models/" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
