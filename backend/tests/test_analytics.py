"""
Automated Unit Tests for EAR, Rolling PERCLOS, and Blink Analytics
"""

import sys
import pytest
import numpy as np
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics.eye_aspect_ratio import calculate_ear_single_eye, EyeTracker
from analytics.perclos import RollingPERCLOS
from analytics.blink_tracker import BlinkTracker

def test_ear_calculation():
    """Validates EAR formula on synthetic geometry."""
    # Open eye: height ~ 10, width ~ 20 -> EAR ~ (10 + 10) / (2 * 20) = 0.5
    open_eye = [
        np.array([0, 10]),   # p1 outer
        np.array([5, 15]),   # p2 top1
        np.array([15, 15]),  # p3 top2
        np.array([20, 10]),  # p4 inner
        np.array([15, 5]),   # p5 bottom2
        np.array([5, 5])     # p6 bottom1
    ]
    ear = calculate_ear_single_eye(open_eye)
    assert 0.45 <= ear <= 0.55

    # Closed eye: height ~ 1, width ~ 20 -> EAR ~ (1 + 1) / 40 = 0.05
    closed_eye = [
        np.array([0, 10]),
        np.array([5, 10.5]),
        np.array([15, 10.5]),
        np.array([20, 10]),
        np.array([15, 9.5]),
        np.array([5, 9.5])
    ]
    ear_closed = calculate_ear_single_eye(closed_eye)
    assert ear_closed < 0.10

def test_rolling_perclos_duration_accumulation():
    """Validates that PERCLOS correctly accumulates duration over rolling window."""
    perclos = RollingPERCLOS(window_seconds=10.0, drowsy_threshold=0.25, min_coverage=0.30)
    
    # Feed 10 seconds of data: 7 seconds open, 3 seconds closed (30% closed -> Drowsy)
    base_ts = 1000.0
    for i in range(70):  # 7 seconds open (at 10 Hz)
        perclos.update(is_closed=False, is_valid=True, timestamp=base_ts + i * 0.1, dt=0.1)
    for i in range(30):  # 3 seconds closed
        perclos.update(is_closed=True, is_valid=True, timestamp=base_ts + 7.0 + i * 0.1, dt=0.1)

    metrics = perclos.get_metrics()
    assert metrics["valid_observation"] is True
    assert 0.28 <= metrics["perclos_ratio"] <= 0.32
    assert metrics["is_drowsy"] is True
    assert metrics["perclos_percent"] >= 28.0

def test_perclos_insufficient_coverage():
    """Validates that PERCLOS marks reading invalid if face is missing for majority of window."""
    perclos = RollingPERCLOS(window_seconds=10.0, min_coverage=0.50)
    # Only 1 second of valid data in a 10s window (10% coverage)
    base_ts = 2000.0
    for i in range(10):
        perclos.update(is_closed=False, is_valid=True, timestamp=base_ts + i * 0.1, dt=0.1)

    metrics = perclos.get_metrics()
    assert metrics["valid_observation"] is False
    assert "Insufficient Coverage" in metrics["status_text"]

def test_blink_tracker_state_machine():
    """Validates discrete blink detection (OPEN -> CLOSED -> REOPENED)."""
    tracker = BlinkTracker(history_window_sec=60.0)
    base_ts = 500.0

    # Start open
    tracker.update(is_closed=False, is_valid=True, timestamp=base_ts)
    assert tracker.total_blinks == 0

    # Eyes close at base_ts + 0.1
    tracker.update(is_closed=True, is_valid=True, timestamp=base_ts + 0.1)
    assert tracker.total_blinks == 0
    assert tracker.current_state == "CLOSING"

    # Eyes reopen 200ms later at base_ts + 0.3 -> valid blink!
    event = tracker.update(is_closed=False, is_valid=True, timestamp=base_ts + 0.3)
    assert tracker.total_blinks == 1
    assert event is not None
    assert event["type"] == "NORMAL_BLINK"
    assert tracker.current_state == "OPEN"
