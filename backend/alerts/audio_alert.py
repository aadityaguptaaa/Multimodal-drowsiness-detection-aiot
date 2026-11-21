"""
Audio Alert System for Driver Drowsiness
Provides non-blocking Windows winsound audible alerting with cooldown protection.
"""

import time
import threading
import sys
from pathlib import Path

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

# Windows-native beep support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AudioAlertSystem:
    """
    Manages audible warnings with cooldown timers, mute toggling, and multi-tier sounds.
    """
    def __init__(self, cooldown_seconds=config.ALERT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.enabled = True
        self.last_alert_time = 0.0
        self.last_alert_level = "NONE"
        self.is_playing = False
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool):
        with self._lock:
            self.enabled = bool(enabled)

    def is_in_cooldown(self) -> bool:
        return (time.time() - self.last_alert_time) < self.cooldown_seconds

    def get_status(self):
        now = time.time()
        remaining_cd = max(0.0, self.cooldown_seconds - (now - self.last_alert_time))
        return {
            "enabled": self.enabled,
            "in_cooldown": remaining_cd > 0.0,
            "cooldown_remaining_sec": round(remaining_cd, 1),
            "last_alert_level": self.last_alert_level,
            "last_alert_time": self.last_alert_time
        }

    def trigger(self, alert_level="WARNING", force=False):
        """
        Triggers an audible alert if enabled and cooldown has elapsed.
        Runs asynchronously in a background daemon thread so it never blocks inference.
        """
        if not self.enabled and not force:
            return False

        with self._lock:
            now = time.time()
            if not force and (now - self.last_alert_time) < self.cooldown_seconds:
                return False  # In cooldown

            self.last_alert_time = now
            self.last_alert_level = alert_level

        thread = threading.Thread(target=self._play_sound_worker, args=(alert_level,), daemon=True)
        thread.start()
        return True

    def _play_sound_worker(self, level):
        try:
            self.is_playing = True
            if HAS_WINSOUND:
                if level == "CRITICAL":
                    # Double high-pitch rapid pulse for critical danger
                    winsound.Beep(config.ALERT_BEEP_FREQ_CRITICAL, 250)
                    time.sleep(0.08)
                    winsound.Beep(config.ALERT_BEEP_FREQ_CRITICAL, 350)
                else:  # WARNING
                    # Single lower-pitch notification tone
                    winsound.Beep(config.ALERT_BEEP_FREQ_WARNING, 300)
            else:
                # Terminal bell fallback on non-Windows environments
                print("\a", end="", flush=True)
        except Exception as e:
            print(f"[AudioAlert Warning] Sound playback skipped: {e}")
        finally:
            self.is_playing = False

    def test_alert(self):
        """Manual test trigger for UI testing."""
        return self.trigger("WARNING", force=True)
