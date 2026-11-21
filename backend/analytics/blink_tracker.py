"""
Blink Tracker and Dynamics Analytics Module
Detects discrete blink events via state machine transitions (OPEN -> CLOSED -> REOPENED),
tracks blinks per minute (BPM), blink duration, and analyzes temporal trends.
"""

import time
from collections import deque
import numpy as np

class BlinkTracker:
    """
    Maintains blink state machine and calculates rate/trend metrics.
    """
    def __init__(self, history_window_sec=60.0, min_blink_duration=0.08, max_blink_duration=0.50):
        self.history_window_sec = history_window_sec
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration

        # State machine: 'OPEN' or 'CLOSING'
        self.current_state = "OPEN"
        self.close_start_time = None
        self.total_blinks = 0

        # Queue of recorded blinks: list of (timestamp, duration_sec)
        self.blink_history = deque()

        # Baseline blink rate (estimated or calibrated)
        self.baseline_bpm = 16.0  # Normal adult average ~12-18 BPM

    def update(self, is_closed, is_valid, timestamp=None):
        """
        Updates blink state machine with new frame observation.
        """
        now = time.time() if timestamp is None else timestamp

        if not is_valid:
            # If face lost during closure, reset state to avoid false long blink
            if self.current_state == "CLOSING":
                self.current_state = "OPEN"
                self.close_start_time = None
            return None

        detected_blink = None

        if self.current_state == "OPEN":
            if is_closed:
                self.current_state = "CLOSING"
                self.close_start_time = now
        elif self.current_state == "CLOSING":
            if not is_closed:
                # Eyes reopened! Check duration to confirm valid blink
                duration = now - self.close_start_time
                if self.min_blink_duration <= duration <= self.max_blink_duration:
                    self.total_blinks += 1
                    detected_blink = {
                        "timestamp": now,
                        "duration_ms": round(duration * 1000.0, 1),
                        "type": "NORMAL_BLINK"
                    }
                    self.blink_history.append((now, duration))
                elif duration > self.max_blink_duration:
                    # Prolonged eye closure (microsleep candidate)
                    self.total_blinks += 1
                    detected_blink = {
                        "timestamp": now,
                        "duration_ms": round(duration * 1000.0, 1),
                        "type": "PROLONGED_CLOSURE"
                    }
                    self.blink_history.append((now, duration))

                self.current_state = "OPEN"
                self.close_start_time = None
            else:
                # Eyes still closed - check if prolonged closure is ongoing
                ongoing_duration = now - self.close_start_time
                if ongoing_duration > 1.2:
                    # Flag ongoing microsleep
                    detected_blink = {
                        "timestamp": now,
                        "duration_ms": round(ongoing_duration * 1000.0, 1),
                        "type": "ONGOING_PROLONGED_CLOSURE"
                    }

        # Purge blinks older than history_window_sec
        cutoff = now - self.history_window_sec
        while self.blink_history and self.blink_history[0][0] < cutoff:
            self.blink_history.popleft()

        return detected_blink

    def get_metrics(self, current_timestamp=None):
        """
        Computes current blink frequency, average duration, and trend.
        """
        now = time.time() if current_timestamp is None else current_timestamp
        recent_count = len(self.blink_history)

        # Calculate BPM scaled to 60 seconds
        bpm = float(recent_count)

        avg_duration_ms = 0.0
        if recent_count > 0:
            durations = [d * 1000.0 for _, d in self.blink_history]
            avg_duration_ms = round(float(np.mean(durations)), 1)

        # Blink Trend Analysis:
        # High BPM (> 28) can indicate fight against fatigue / fluttering eyelids
        # Low BPM with long duration indicates drowsy staring or impending microsleep
        trend = "NORMAL"
        if bpm > 26.0:
            trend = "RAPID_FLUTTERING (Fatigue Compensation)"
        elif bpm < 6.0 and recent_count > 0:
            trend = "DEPRESSED_RATE (Staring / Drowsiness)"
        elif avg_duration_ms > 350.0:
            trend = "SLUGGISH_BLINKS (Slow Reopening)"

        return {
            "total_blinks": self.total_blinks,
            "recent_blinks_window": recent_count,
            "blinks_per_minute": round(bpm, 1),
            "average_duration_ms": avg_duration_ms,
            "blink_trend": trend,
            "history_window_sec": self.history_window_sec
        }

    def reset(self):
        self.current_state = "OPEN"
        self.close_start_time = None
        self.total_blinks = 0
        self.blink_history.clear()
