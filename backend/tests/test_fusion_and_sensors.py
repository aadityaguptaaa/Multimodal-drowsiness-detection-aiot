"""
Automated Unit Tests for Sensors, Multi-Signal Fusion, and Audio Alert Cooldown
"""

import time
import sys
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sensors.simulated_mpu6050 import SimulatedMPU6050
from sensors.real_mpu6050 import RealMPU6050
from fusion.drowsiness_fusion import DrowsinessFusionEngine
from alerts.audio_alert import AudioAlertSystem

def test_sensor_simulator_states():
    """Tests that SimulatedMPU6050 correctly models NORMAL, HEAD_NOD, HEAD_TILT, and DISCONNECTED."""
    sim = SimulatedMPU6050()
    
    # Normal state
    sim.set_state("NORMAL")
    r_norm = sim.read()
    assert r_norm.is_valid is True
    assert r_norm.mode in ["I2C_ONLINE", "SIMULATED"]
    assert -10.0 <= r_norm.pitch <= 10.0

    # Head nod state (downward pitch < -15°)
    sim.set_state("HEAD_NOD")
    r_nod = sim.read()
    assert r_nod.is_valid is True
    assert r_nod.pitch <= -15.0

    # Head tilt state (lateral roll > 20°)
    sim.set_state("HEAD_TILT")
    r_tilt = sim.read()
    assert r_tilt.is_valid is True
    assert abs(r_tilt.roll) >= 20.0

    # Disconnected state
    sim.set_state("SENSOR_DISCONNECTED")
    r_disc = sim.read()
    assert r_disc.is_valid is False

def test_real_sensor_graceful_handling():
    """Tests that RealMPU6050 safely returns valid=False without throwing exceptions when hardware is unplugged."""
    real = RealMPU6050(port="COM99_UNPLUGGED", auto_start=False)
    # When targeting an unplugged port and bridge is not on COM99
    assert real.get_mode() == "REAL"
    reading = real.bridge.get_latest_reading()
    assert reading.is_valid is False
    assert reading.is_connected is False

def test_fusion_decision_logic():
    """Validates explainable multi-signal fusion rules."""
    engine = DrowsinessFusionEngine()

    # Case 1: All nominal -> NORMAL
    res_norm = engine.fuse(
        cnn_result={"is_valid": True, "model_loaded": True, "probability": 0.15},
        perclos_result={"valid_observation": True, "perclos_ratio": 0.05},
        blink_result={"blinks_per_minute": 15.0, "average_duration_ms": 180.0, "blink_trend": "NORMAL"},
        sensor_reading={"is_valid": True, "pitch": 0.0, "roll": 0.0, "mode": "SIMULATED"}
    )
    assert res_norm["overall_state"] == "NORMAL"
    assert res_norm["composite_score"] < 40.0

    # Case 2: High CNN + High PERCLOS -> CRITICAL
    res_crit = engine.fuse(
        cnn_result={"is_valid": True, "model_loaded": True, "probability": 0.88},
        perclos_result={"valid_observation": True, "perclos_ratio": 0.35},
        blink_result={"blinks_per_minute": 5.0, "average_duration_ms": 420.0, "blink_trend": "SLUGGISH_BLINKS"},
        sensor_reading={"is_valid": True, "pitch": -28.0, "roll": 0.0, "mode": "SIMULATED"}
    )
    assert res_crit["overall_state"] == "CRITICAL"
    assert len(res_crit["contributing_factors"]) >= 2
    assert any("CNN" in f for f in res_crit["contributing_factors"])
    assert any("PERCLOS" in f for f in res_crit["contributing_factors"])

def test_audio_alert_cooldown():
    """Validates that rapid repeated triggers are prevented by cooldown."""
    alert = AudioAlertSystem(cooldown_seconds=1.5)
    
    # First trigger should succeed
    t1 = alert.trigger("WARNING")
    assert t1 is True
    assert alert.is_in_cooldown() is True

    # Immediate second trigger should be blocked by cooldown
    t2 = alert.trigger("WARNING")
    assert t2 is False

    # Force test bypasses cooldown
    t3 = alert.trigger("WARNING", force=True)
    assert t3 is True
