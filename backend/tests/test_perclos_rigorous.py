"""
Rigorous Unit Tests for PERCLOS Calculation & Sustained Closure Detection
Validates:
1. Timestamp-based elapsed duration accumulation (variable dt)
2. Sustained closure (> 1.5s) triggering microsleep detection vs momentary blinks (200ms)
3. Missing-face observations not treated as open eyes (reduce coverage percentage)
4. Long tracking interruption (> 3.0s gap) invalidating / resetting stale buffer
"""

import sys
import time
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics.perclos import RollingPERCLOS


def test_controlled_perclos_accumulation():
    """Validates exact mathematical PERCLOS calculation over variable frame deltas."""
    tracker = RollingPERCLOS(
        window_seconds=10.0,
        drowsy_threshold=0.25,
        min_coverage=0.50
    )
    base_t = 1000.0

    # 6 seconds open eyes (60 frames at 100ms dt)
    for i in range(60):
        tracker.update(is_closed=False, is_valid=True, timestamp=base_t + i * 0.1, dt=0.1)

    # 4 seconds closed eyes (40 frames at 100ms dt)
    for i in range(40):
        tracker.update(is_closed=True, is_valid=True, timestamp=base_t + 6.0 + i * 0.1, dt=0.1)

    m = tracker.get_metrics()
    assert m["valid_observation"] is True
    # Exactly 4.0s closed out of 10.0s valid observation = 40.0%
    assert 39.0 <= m["perclos_percent"] <= 41.0
    assert m["is_drowsy"] is True
    assert m["closed_duration_sec"] == 4.0
    assert m["open_duration_sec"] == 6.0


def test_sustained_closure_vs_short_blinks():
    """Validates that a 200ms normal blink does NOT trigger microsleep, but >= 1.5s closure does."""
    tracker = RollingPERCLOS(
        window_seconds=10.0,
        sustained_closure_threshold_sec=1.5
    )
    t = 2000.0

    # Scenario A: Normal blink (open 1s, closed 200ms, open 1s)
    tracker.update(is_closed=False, is_valid=True, timestamp=t, dt=1.0)
    tracker.update(is_closed=True, is_valid=True, timestamp=t + 1.1, dt=0.1)
    tracker.update(is_closed=True, is_valid=True, timestamp=t + 1.2, dt=0.1)
    tracker.update(is_closed=False, is_valid=True, timestamp=t + 1.3, dt=0.1)

    m1 = tracker.get_metrics()
    assert m1["is_sustained_closure"] is False
    assert m1["sustained_closure_sec"] == 0.0

    # Scenario B: Sustained continuous closure for 1.8 seconds
    for i in range(18):
        tracker.update(is_closed=True, is_valid=True, timestamp=t + 2.0 + i * 0.1, dt=0.1)

    m2 = tracker.get_metrics()
    assert m2["is_sustained_closure"] is True
    assert m2["sustained_closure_sec"] >= 1.5
    assert m2["is_drowsy"] is True
    assert "MICROSLEEP" in m2["status_text"]


def test_missing_face_not_treated_as_open_eyes():
    """Validates that face occlusion reduces observation coverage rather than diluting closure with fake open eyes."""
    tracker = RollingPERCLOS(window_seconds=10.0, min_coverage=0.50)
    t = 3000.0

    # 2 seconds valid closed eyes
    for i in range(20):
        tracker.update(is_closed=True, is_valid=True, timestamp=t + i * 0.1, dt=0.1)

    # 8 seconds of MISSING face (e.g. driver turned away / camera occluded)
    for i in range(80):
        tracker.update(is_closed=False, is_valid=False, timestamp=t + 2.0 + i * 0.1, dt=0.1)

    m = tracker.get_metrics()
    # Total valid time is only 2.0s out of 10s window (20% coverage < 50% min_coverage)
    assert m["valid_observation"] is False
    assert m["observation_coverage"] <= 0.25
    assert "Insufficient Coverage" in m["status_text"]
    # Open duration must be 0, NOT 8 seconds!
    assert m["open_duration_sec"] == 0.0


def test_long_tracking_gap_resets_buffer():
    """Validates that a tracking interruption > max_gap_seconds (3.0s) clears stale history."""
    tracker = RollingPERCLOS(window_seconds=10.0, max_gap_seconds=3.0)
    t = 4000.0

    # Add 5 seconds of closed eyes
    for i in range(50):
        tracker.update(is_closed=True, is_valid=True, timestamp=t + i * 0.1, dt=0.1)

    # Long tracking gap of 5.0 seconds
    t_resume = t + 5.0 + 5.0
    tracker.update(is_closed=False, is_valid=True, timestamp=t_resume, dt=0.1)

    # Buffer should have been cleared; only the new resume frame should be present
    assert len(tracker.buffer) == 1
    m = tracker.get_metrics()
    assert m["closed_duration_sec"] == 0.0
    assert m["open_duration_sec"] == 0.1
