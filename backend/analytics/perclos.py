"""
Dedicated PERCLOS (Percentage of Eye Closure) Calculation Module
Implements standard temporal accumulation of eye-closure durations:
    PERCLOS = (Duration classified as closed / Total valid observation time) * 100

Key Engineering Principles:
- Computes duration from explicit timestamps and delta time (dt), not assuming constant frame rate.
- Tracks open-eye and closed-eye durations separately.
- Invalid observations (missing face, occlusion) never count as open eyes; they reduce valid coverage.
- Tracks sustained eye closure duration to differentiate microsleep from normal blinks.
- Resets/invalidates measurement if tracking is interrupted longer than MAX_TRACKING_GAP_SECONDS.
- Does not assert hardcoded universal validity; alert thresholds are configurable and clearly documented.
"""

import time
from collections import deque
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


class RollingPERCLOS:
    """
    Maintains a rolling time window of timestamped eye-state observations.
    Computes genuine PERCLOS and sustained eye-closure durations.
    """
    def __init__(
        self,
        window_seconds: float = config.PERCLOS_WINDOW_SECONDS,
        drowsy_threshold: float = config.PERCLOS_DROWSY_THRESHOLD,
        warning_threshold: float = config.PERCLOS_WARNING_THRESHOLD,
        min_coverage: float = config.MIN_OBSERVATION_COVERAGE,
        sustained_closure_threshold_sec: float = config.SUSTAINED_CLOSURE_DROWSY_THRESHOLD_SEC,
        max_gap_seconds: float = config.MAX_TRACKING_GAP_SECONDS
    ):
        self.window_seconds = float(window_seconds)
        self.drowsy_threshold = float(drowsy_threshold)
        self.warning_threshold = float(warning_threshold)
        self.min_coverage = float(min_coverage)
        self.sustained_closure_threshold_sec = float(sustained_closure_threshold_sec)
        self.max_gap_seconds = float(max_gap_seconds)

        # Buffer entries: (timestamp, is_closed, is_valid, dt)
        self.buffer = deque()

        # Tracking state
        self._last_timestamp: Optional[float] = None
        self._current_closure_start: Optional[float] = None
        self._current_sustained_closure_sec: float = 0.0

    def update(
        self,
        is_closed: bool,
        is_valid: bool,
        timestamp: Optional[float] = None,
        dt: Optional[float] = None
    ):
        """
        Records a new timestamped observation frame.

        Args:
            is_closed: True if eye classifier/EAR indicates closed state.
            is_valid: True only if a face and eyes were actively tracked.
            timestamp: Unix epoch timestamp.
            dt: Optional elapsed frame duration. If None, derived from timestamp delta.
        """
        now = time.time() if timestamp is None else float(timestamp)

        # 1. Check for long tracking gap: reset window if tracking interrupted too long
        if self._last_timestamp is not None:
            gap = now - self._last_timestamp
            if gap > self.max_gap_seconds:
                # Driver face was lost for too long -> invalidate stale buffer
                self.buffer.clear()
                self._current_closure_start = None
                self._current_sustained_closure_sec = 0.0

        # 2. Derive delta time
        if dt is None:
            if self._last_timestamp is not None:
                derived_dt = max(0.001, min(1.0, now - self._last_timestamp))
            else:
                derived_dt = 0.033  # Initial ~30 FPS estimate
        else:
            derived_dt = max(0.0001, float(dt))

        self._last_timestamp = now

        # 3. Track sustained continuous closure (for microsleep detection)
        if is_valid and is_closed:
            if self._current_closure_start is None:
                self._current_closure_start = now
            self._current_sustained_closure_sec = round(now - self._current_closure_start, 2)
        else:
            self._current_closure_start = None
            self._current_sustained_closure_sec = 0.0

        # 4. Append observation to rolling buffer
        self.buffer.append((now, bool(is_closed), bool(is_valid), float(derived_dt)))

        # 5. Evict samples outside the rolling window
        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculates PERCLOS metrics for the current valid window.
        Formula:
            PERCLOS = (Duration classified as closed) / (Total valid observation duration) * 100
        """
        if not self.buffer:
            return {
                "perclos_ratio": 0.0,
                "perclos_percent": 0.0,
                "observation_coverage": 0.0,
                "coverage_percent": 0.0,
                "valid_observation": False,
                "is_drowsy": False,
                "is_warning": False,
                "is_sustained_closure": False,
                "sustained_closure_sec": 0.0,
                "sustained_closure_threshold_sec": self.sustained_closure_threshold_sec,
                "window_duration_sec": 0.0,
                "valid_duration_sec": 0.0,
                "closed_duration_sec": 0.0,
                "open_duration_sec": 0.0,
                "drowsy_threshold_used": self.drowsy_threshold,
                "warning_threshold_used": self.warning_threshold,
                "status_text": "No Data in Window"
            }

        total_window_span = max(0.001, self.buffer[-1][0] - self.buffer[0][0])

        valid_duration = 0.0
        closed_duration = 0.0
        open_duration = 0.0

        for ts, is_closed, is_valid, frame_dt in self.buffer:
            if is_valid:
                valid_duration += frame_dt
                if is_closed:
                    closed_duration += frame_dt
                else:
                    open_duration += frame_dt

        # Coverage represents fraction of window time where face tracking was valid
        coverage = min(1.0, valid_duration / max(0.1, self.window_seconds))

        # Check for sustained continuous closure (immediate high risk)
        is_sustained_closure = self._current_sustained_closure_sec >= self.sustained_closure_threshold_sec

        # Insufficient coverage check
        if coverage < self.min_coverage or valid_duration < 1.0:
            status_text = (
                f"MICROSLEEP DETECTED ({self._current_sustained_closure_sec:.1f}s closed)"
                if is_sustained_closure
                else f"Insufficient Coverage ({int(coverage * 100)}% < {int(self.min_coverage * 100)}%)"
            )
            return {
                "perclos_ratio": 0.0,
                "perclos_percent": 0.0,
                "observation_coverage": round(coverage, 2),
                "coverage_percent": round(coverage * 100.0, 1),
                "valid_observation": False,
                "is_drowsy": is_sustained_closure,  # Sustained closure triggers even if coverage is still accumulating
                "is_warning": False,
                "is_sustained_closure": is_sustained_closure,
                "sustained_closure_sec": self._current_sustained_closure_sec,
                "sustained_closure_threshold_sec": self.sustained_closure_threshold_sec,
                "window_duration_sec": round(total_window_span, 1),
                "valid_duration_sec": round(valid_duration, 2),
                "closed_duration_sec": round(closed_duration, 2),
                "open_duration_sec": round(open_duration, 2),
                "drowsy_threshold_used": self.drowsy_threshold,
                "warning_threshold_used": self.warning_threshold,
                "status_text": status_text
            }

        # Calculate genuine PERCLOS ratio and percentage
        perclos_ratio = min(1.0, max(0.0, closed_duration / valid_duration))
        perclos_percent = round(perclos_ratio * 100.0, 1)

        is_drowsy = (perclos_ratio >= self.drowsy_threshold) or is_sustained_closure
        is_warning = (perclos_ratio >= self.warning_threshold) and not is_drowsy

        if is_sustained_closure:
            status_text = f"MICROSLEEP DETECTED ({self._current_sustained_closure_sec}s closed)"
        elif is_drowsy:
            status_text = f"DROWSY (PERCLOS {perclos_percent}%)"
        elif is_warning:
            status_text = f"FATIGUE WARNING (PERCLOS {perclos_percent}%)"
        else:
            status_text = f"NORMAL (PERCLOS {perclos_percent}%)"

        return {
            "perclos_ratio": round(perclos_ratio, 3),
            "perclos_percent": perclos_percent,
            "observation_coverage": round(coverage, 2),
            "coverage_percent": round(coverage * 100.0, 1),
            "valid_observation": True,
            "is_drowsy": is_drowsy,
            "is_warning": is_warning,
            "is_sustained_closure": is_sustained_closure,
            "sustained_closure_sec": self._current_sustained_closure_sec,
            "sustained_closure_threshold_sec": self.sustained_closure_threshold_sec,
            "window_duration_sec": round(total_window_span, 1),
            "valid_duration_sec": round(valid_duration, 2),
            "closed_duration_sec": round(closed_duration, 2),
            "open_duration_sec": round(open_duration, 2),
            "drowsy_threshold_used": self.drowsy_threshold,
            "warning_threshold_used": self.warning_threshold,
            "status_text": status_text
        }

    def reset(self):
        """Clears all historical observations in buffer."""
        self.buffer.clear()
        self._last_timestamp = None
        self._current_closure_start = None
        self._current_sustained_closure_sec = 0.0
