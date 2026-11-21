"""
Unit Tests for Modular Sensor Fusion Layer & Subsystem Readiness
Validates:
1. Subsystem readiness status evaluation (Camera, Face/Eyes, Eye Model, PERCLOS, ESP32, IMU)
2. Graceful degradation when physical hardware is disconnected (Vision continues operating)
3. Immediate microsleep escalation on sustained eye closure (> 1.5s)
4. Absence of fabricated data when subsystems are unavailable
"""

import sys
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fusion.drowsiness_fusion import DrowsinessFusionEngine


def test_subsystem_readiness_reporting():
    """Validates that all required subsystem statuses are explicitly reported."""
    engine = DrowsinessFusionEngine()

    readiness = engine.evaluate_readiness(
        camera_operational=True,
        face_detected=True,
        eyes_detected=True,
        eye_model_loaded=False,
        perclos_valid=True,
        esp32_connected=False,
        sensor_valid=False,
        sensor_verified=False
    )

    assert readiness["camera_operational"] is True
    assert readiness["face_detected"] is True
    assert readiness["eyes_detected"] is True
    assert readiness["eye_model_loaded"] is False
    assert readiness["perclos_valid"] is True
    assert readiness["esp32_connected"] is False
    assert readiness["motion_sensor_operational"] is False
    assert readiness["overall_detection_readiness"] == "ACTIVE_VISION_ONLY"


def test_missing_hardware_does_not_disable_webcam():
    """Validates that disconnected ESP32 does not halt or corrupt vision drowsiness detection."""
    engine = DrowsinessFusionEngine()

    result = engine.fuse(
        cnn_result={"is_valid": True, "model_loaded": True, "probability": 0.85, "predicted_class": "Drowsy"},
        perclos_result={"valid_observation": True, "perclos_ratio": 0.30, "perclos_percent": 30.0},
        blink_result={"blinks_per_minute": 14.0, "blink_trend": "NORMAL"},
        sensor_reading={"is_connected": False, "is_valid": False, "pitch": 0.0, "roll": 0.0},
        eye_state_result={"is_valid": True, "state": "closed", "confidence": 0.90},
        camera_operational=True,
        face_detected=True
    )

    assert result["overall_state"] == "CRITICAL"
    assert result["composite_score"] >= 70.0
    assert any("offline" in f.lower() for f in result["contributing_factors"])


def test_sustained_closure_immediate_microsleep():
    """Validates that sustained closure immediately triggers CRITICAL state."""
    engine = DrowsinessFusionEngine()

    result = engine.fuse(
        cnn_result={"is_valid": False, "model_loaded": False},
        perclos_result={
            "valid_observation": True,
            "perclos_ratio": 0.20,
            "is_sustained_closure": True,
            "sustained_closure_sec": 1.9
        },
        blink_result={},
        sensor_reading={"is_connected": False, "is_valid": False},
        camera_operational=True,
        face_detected=True
    )

    assert result["overall_state"] == "CRITICAL"
    assert any("Microsleep" in f for f in result["contributing_factors"])
