"""
Central Configuration for AIoT Driver Drowsiness Detection System
Preserves specifications from project presentation:
- Driver Drowsiness Dataset (DDD) derived from RLDD (41,790+ images)
- 2 Classes: Drowsy (1) and Non Drowsy (0)
- MobileNetV2 transfer learning (224x224 input)
- Rolling PERCLOS (10-second default window)
- Blink frequency & duration tracking
- Sensor fusion (CNN + PERCLOS + Blinks + Head Pitch/Roll)
- Audio alert with cooldown
- Target budget: ₹3,000–₹4,000
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
# Supports both workspace relative path and direct DDD path
DATASET_DIR = PROJECT_ROOT / "Driver Drowsiness Dataset (DDD)"
if not DATASET_DIR.exists():
    DATASET_DIR = PROJECT_ROOT / "dataset"

CLASS_NAMES = ["Non Drowsy", "Drowsy"]
IMAGE_SIZE = (224, 224)  # MobileNetV2 default
INPUT_SHAPE = (224, 224, 3)
ORIGINAL_DATASET_SIZE = (227, 227)

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model Artifact Paths
MODEL_PATH = MODELS_DIR / "drowsiness_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METADATA_PATH = MODELS_DIR / "training_metadata.json"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.png"
DATASET_REPORT_PATH = MODELS_DIR / "dataset_inspection_report.json"

# Eye-State Model Artifact Paths (open vs closed)
EYE_DATASET_DIR = PROJECT_ROOT / "dataset"
EYE_MODEL_PATH = MODELS_DIR / "eye_state_model.keras"
EYE_METADATA_PATH = MODELS_DIR / "eye_training_metadata.json"
EYE_EVALUATION_REPORT_PATH = MODELS_DIR / "eye_evaluation_report.json"
EYE_CONFUSION_MATRIX_PATH = MODELS_DIR / "eye_confusion_matrix.png"
EYE_CLASS_NAMES = ["open", "closed"]
EYE_IMAGE_SIZE = (64, 64)

# Eye Aspect Ratio (EAR) & Blink Analytics
EAR_THRESHOLD = 0.22         # Eye considered closed when EAR < threshold
EAR_CONSEC_FRAMES = 2        # Consecutive frames for a blink candidate
BLINK_MIN_DURATION = 0.08    # Min duration for a valid blink (seconds)
BLINK_MAX_DURATION = 0.45    # Max duration for a valid normal blink (seconds)

# PERCLOS Parameters (Configurable Engineering Parameters)
PERCLOS_WINDOW_SECONDS = 10.0  # Rolling observation window
PERCLOS_DROWSY_THRESHOLD = 0.25 # >= 25% eye closure in window indicates drowsiness
PERCLOS_WARNING_THRESHOLD = 0.15 # >= 15% eye closure indicates fatigue build-up
MIN_OBSERVATION_COVERAGE = 0.50 # Min valid frames required in window (50%)
SUSTAINED_CLOSURE_DROWSY_THRESHOLD_SEC = 1.5 # Consecutive eye closure >= 1.5s flags microsleep
MAX_TRACKING_GAP_SECONDS = 3.0 # Missing face tracking gap > 3.0s resets / invalidates window

# Temporal Smoothing for Live Inference
PREDICTION_SMOOTHING_WINDOW = 5 # Rolling frame average for CNN confidence
CNN_DROWSY_THRESHOLD = 0.60

# Physical ESP32 Hardware & Serial Bridge Configuration
SERIAL_PORT = os.environ.get("ESP32_PORT", "COM9")
SERIAL_BAUDRATE = int(os.environ.get("ESP32_BAUDRATE", 115200))
SERIAL_TIMEOUT_SEC = 1.0
SENSOR_STALE_TIMEOUT_SEC = 2.0    # Mark sensor data stale after 2s without serial update

# MPU6050 & Head Pose Thresholds (Degrees)
HEAD_NOD_PITCH_THRESHOLD = -15.0  # Head nodding downward (negative pitch)
HEAD_TILT_ROLL_THRESHOLD = 25.0   # Extreme head tilt left/right (absolute roll)

# Sensor Fusion Weights (Configurable Engineering Parameters)
FUSION_WEIGHT_CNN = 0.45
FUSION_WEIGHT_PERCLOS = 0.35
FUSION_WEIGHT_BLINK = 0.10
FUSION_WEIGHT_HEAD = 0.10

# Audio Alert Settings
ALERT_COOLDOWN_SECONDS = 3.5
ALERT_BEEP_FREQ_WARNING = 1200  # Hz
ALERT_BEEP_FREQ_CRITICAL = 2200 # Hz
ALERT_BEEP_DURATION_MS = 400   # ms

# Server & Network Configuration
API_HOST = "127.0.0.1"
API_PORT = 5000
FRONTEND_PORT = 5173
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Target Project Budget (Student AIoT Solution)
TARGET_BUDGET_INR = "₹3,000–₹4,000"
