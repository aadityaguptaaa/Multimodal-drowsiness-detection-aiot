"""
Unit Tests for ESP32 Serial Telemetry Bridge & Parser
Validates:
1. Parsing of valid ESP32 JSON telemetry packets (ax, ay, az, gx, gy, gz, who_am_i)
2. Graceful rejection of malformed or corrupted serial packets
3. Sensor identity verification (WHO_AM_I register 0x68)
4. Staleness watchdog (data older than 2.0s flagged is_valid=False)
5. Disconnected hardware behavior (never generates synthetic numbers)
"""

import sys
import time
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sensors.esp32_bridge import ESP32SerialBridge
from sensors.base import SensorReading


def test_serial_telemetry_valid_json_parsing():
    """Validates parser on standard ESP32 MPU-6050 JSON payload."""
    bridge = ESP32SerialBridge(port="COM99_NONEXISTENT", auto_start=False)

    valid_json = (
        '{"verified":true,"chip":"MPU-6050","who_am_i":"0x68",'
        '"ax":0.25,"ay":-0.10,"az":9.75,'
        '"gx":0.5,"gy":-0.2,"gz":0.1,'
        '"pitch":-1.5,"roll":0.6,"yaw":0.0,"ts_ms":123456}'
    )

    result = bridge._parse_telemetry_line(valid_json)
    assert result is not None
    metrics, meta = result

    assert metrics["ax"] == 0.25
    assert metrics["ay"] == -0.10
    assert metrics["az"] == 9.75
    assert metrics["gx"] == 0.5
    assert metrics["pitch"] == -1.5
    assert metrics["roll"] == 0.6
    assert meta["chip_verified"] is True
    assert "MPU-6050" in meta["chip_id"]


def test_serial_telemetry_malformed_string_rejection():
    """Validates that garbage, noise, or partial packets are safely dropped."""
    bridge = ESP32SerialBridge(port="COM99_NONEXISTENT", auto_start=False)

    assert bridge._parse_telemetry_line("") is None
    assert bridge._parse_telemetry_line("GARBAGE_NOISE_LINE_123") is None
    assert bridge._parse_telemetry_line('{"ax": 0.5, "ay": ') is None # Incomplete JSON
    # Physically impossible acceleration (> 100 m/s^2) should be rejected
    assert bridge._parse_telemetry_line('{"ax": 99999.0, "ay": 0.0, "az": 0.0}') is None


def test_chip_verification_flags():
    """Validates that unverified chip IDs are not flagged as authentic MPU-6050."""
    bridge = ESP32SerialBridge(port="COM99_NONEXISTENT", auto_start=False)

    unverified_json = '{"verified":false,"who_am_i":"0xFF","ax":0.0,"ay":0.0,"az":9.8}'
    result = bridge._parse_telemetry_line(unverified_json)
    assert result is not None
    metrics, meta = result
    assert meta["chip_verified"] is False


def test_staleness_detection():
    """Validates that a reading older than stale_timeout_sec (2.0s) has is_valid set to False."""
    bridge = ESP32SerialBridge(port="COM99_NONEXISTENT", auto_start=False)

    # Manually inject an old timestamp into internal state
    old_time = time.time() - 3.5  # 3.5 seconds ago (> 2.0s timeout)
    with bridge._lock:
        bridge._latest_reading = SensorReading(
            pitch=-2.0,
            roll=1.0,
            yaw=0.0,
            ax=0.0,
            ay=0.0,
            az=9.8,
            timestamp=old_time,
            is_valid=True,
            is_connected=True,
            mode="REAL",
            port="COM9",
            chip_verified=True,
            chip_id="MPU-6050 (0x68)",
            state_description="Normal",
            data_age_sec=0.0
        )

    reading = bridge.get_latest_reading()
    assert reading.is_valid is False
    assert reading.data_age_sec >= 3.0
    assert "Stale" in reading.state_description


def test_disconnected_hardware_safe_state():
    """Validates that bridge reports disconnected=False without throwing unhandled errors."""
    bridge = ESP32SerialBridge(port="COM99_NONEXISTENT", auto_start=False)
    assert bridge.is_connected() is False

    reading = bridge.get_latest_reading()
    assert reading.is_connected is False
    assert reading.is_valid is False
    assert reading.ax == 0.0
    assert reading.ay == 0.0
    assert reading.az == 0.0
