"""
ESP32 Hardware Serial Telemetry Bridge with pyserial
Connects to physical ESP32 development board via USB Serial (COM9 default).
Reads real accelerometer (ax, ay, az) and gyroscope (gx, gy, gz) readings.
Performs sensor identity verification (WHO_AM_I register) and staleness detection.
Never generates synthetic data when disconnected.
"""

import sys
import time
import json
import math
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from sensors.base import SensorReading

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False
    serial = None


class ESP32SerialBridge:
    """
    Thread-safe USB serial bridge for ESP32 + MPU6050 telemetry.
    Maintains a resilient background listener thread with auto-reconnection.
    """
    def __init__(
        self,
        port: str = config.SERIAL_PORT,
        baudrate: int = config.SERIAL_BAUDRATE,
        timeout: float = config.SERIAL_TIMEOUT_SEC,
        stale_timeout_sec: float = config.SENSOR_STALE_TIMEOUT_SEC,
        auto_start: bool = True
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.stale_timeout_sec = stale_timeout_sec

        self._serial_conn = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Connection & Telemetry State
        self._connected = False
        self._connection_error = "Not connected yet"
        self._chip_verified = False
        self._chip_id = "UNVERIFIED"
        self._last_reading_time = 0.0
        self._consecutive_parse_errors = 0
        self._total_frames_received = 0

        # Latest validated measurement
        self._latest_reading = SensorReading(
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            ax=0.0,
            ay=0.0,
            az=0.0,
            gx=0.0,
            gy=0.0,
            gz=0.0,
            timestamp=0.0,
            is_valid=False,
            is_connected=False,
            mode="REAL",
            port=self.port,
            chip_verified=False,
            chip_id="UNVERIFIED",
            state_description="ESP32 Serial Link Inactive",
            error_message="ESP32 hardware not initialized",
            data_age_sec=0.0
        )

        if auto_start:
            self.start()

    @staticmethod
    def list_available_ports() -> list:
        """Returns list of detected serial ports on the host."""
        if not HAS_PYSERIAL or serial is None:
            return []
        try:
            return [{"port": p.device, "description": p.description, "hwid": p.hwid}
                    for p in serial.tools.list_ports.comports()]
        except Exception:
            return []

    def start(self):
        """Starts background serial reader thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="ESP32-SerialBridge")
            self._thread.start()

    def stop(self):
        """Stops background serial reader."""
        self._is_running = False
        with self._lock:
            if self._serial_conn:
                try:
                    self._serial_conn.close()
                except Exception:
                    pass
                self._serial_conn = None
            self._connected = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def set_port(self, port_name: str, baudrate: Optional[int] = None) -> bool:
        """Dynamically reconfigures target serial port and reconnects."""
        self.stop()
        self.port = port_name
        if baudrate:
            self.baudrate = baudrate
        self.start()
        return True

    def is_connected(self) -> bool:
        """Returns True if serial port is currently open and healthy."""
        return self._connected and (self._serial_conn is not None)

    def get_latest_reading(self) -> SensorReading:
        """
        Returns the latest SensorReading.
        Checks for staleness: if data is older than stale_timeout_sec,
        marks is_valid=False.
        """
        now = time.time()
        with self._lock:
            reading = SensorReading(**self._latest_reading.to_dict())

        # Check freshness
        if reading.timestamp > 0:
            reading.data_age_sec = round(now - reading.timestamp, 2)
            if reading.data_age_sec > self.stale_timeout_sec:
                reading.is_valid = False
                reading.state_description = f"Stale Sensor Data ({reading.data_age_sec}s > {self.stale_timeout_sec}s)"
                reading.error_message = "No fresh serial packets received within timeout"
        else:
            reading.is_valid = False
            reading.data_age_sec = 0.0

        return reading

    def get_status(self) -> Dict[str, Any]:
        """Provides diagnostic health information about the serial bridge."""
        reading = self.get_latest_reading()
        return {
            "pyserial_installed": HAS_PYSERIAL,
            "port": self.port,
            "baudrate": self.baudrate,
            "connected": self.is_connected(),
            "connection_error": self._connection_error,
            "chip_verified": self._chip_verified,
            "chip_id": self._chip_id,
            "total_frames_received": self._total_frames_received,
            "data_age_sec": reading.data_age_sec,
            "is_valid": reading.is_valid,
            "available_ports": self.list_available_ports()
        }

    def _attempt_connection(self) -> bool:
        """Attempts to open the serial port."""
        if not HAS_PYSERIAL:
            self._connection_error = "pyserial library is not installed in the Python environment"
            self._connected = False
            return False

        try:
            self._serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            # Flush existing buffer
            self._serial_conn.reset_input_buffer()
            self._connected = True
            self._connection_error = ""
            print(f"[ESP32 Bridge] Successfully opened serial port {self.port} at {self.baudrate} baud.")
            return True
        except serial.SerialException as e:
            self._serial_conn = None
            self._connected = False
            self._connection_error = f"Serial port {self.port} unavailable: {str(e)}"
            return False
        except Exception as e:
            self._serial_conn = None
            self._connected = False
            self._connection_error = f"Unexpected error opening {self.port}: {str(e)}"
            return False

    def _parse_telemetry_line(self, line_str: str) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        """
        Parses and validates incoming raw serial strings.
        Supports both JSON formatting and comma-separated key=value pairs.
        """
        line_str = line_str.strip()
        if not line_str:
            return None

        data: Dict[str, Any] = {}

        # 1. Attempt JSON parsing (preferred standard)
        if line_str.startswith("{") and line_str.endswith("}"):
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError:
                return None
        # 2. Fallback key=value or CSV parsing
        elif "=" in line_str:
            try:
                parts = line_str.replace(";", ",").split(",")
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        data[k.strip().lower()] = v.strip()
            except Exception:
                return None
        else:
            return None

        # Extract axes (normalize keys)
        def get_float(keys, default=0.0):
            for k in keys:
                if k in data:
                    try:
                        return float(data[k])
                    except (ValueError, TypeError):
                        pass
            return default

        ax = get_float(["ax", "acc_x", "accel_x"])
        ay = get_float(["ay", "acc_y", "accel_y"])
        az = get_float(["az", "acc_z", "accel_z"])
        gx = get_float(["gx", "gyro_x", "gyr_x"])
        gy = get_float(["gy", "gyro_y", "gyr_y"])
        gz = get_float(["gz", "gyro_z", "gyr_z"])

        # Check for sensor identity
        who_am_i = str(data.get("who_am_i", data.get("chip_id", ""))).strip().lower()
        verified = bool(data.get("verified", False)) or ("0x68" in who_am_i) or ("104" in who_am_i)

        chip_name = "MPU-6050 (0x68)" if verified else (who_am_i or "UNVERIFIED")

        # Compute derived pitch and roll if not provided directly
        pitch = get_float(["pitch", "head_pitch"], None)
        roll = get_float(["roll", "head_roll"], None)
        yaw = get_float(["yaw", "head_yaw"], 0.0)

        # Trigonometric tilt estimation from raw accelerometer if pitch/roll absent
        if pitch is None or roll is None:
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 0.01:
                calc_pitch = round(math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az))), 1)
                calc_roll = round(math.degrees(math.atan2(ay, az)), 1)
            else:
                calc_pitch, calc_roll = 0.0, 0.0
            pitch = pitch if pitch is not None else calc_pitch
            roll = roll if roll is not None else calc_roll

        # Sanity check physics bounds (e.g. realistic vehicular acceleration and angular rate)
        # Accel should not exceed +- 8g (~78 m/s^2), Gyro +- 1000 deg/s
        if not (-80.0 <= ax <= 80.0 and -80.0 <= ay <= 80.0 and -80.0 <= az <= 80.0):
            return None
        if not (-1000.0 <= gx <= 1000.0 and -1000.0 <= gy <= 1000.0 and -1000.0 <= gz <= 1000.0):
            return None

        metrics = {
            "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
            "gx": round(gx, 2), "gy": round(gy, 2), "gz": round(gz, 2),
            "pitch": round(pitch, 1), "roll": round(roll, 1), "yaw": round(yaw, 1)
        }
        meta = {
            "chip_verified": verified,
            "chip_id": chip_name
        }
        return metrics, meta

    def _reader_loop(self):
        """Continuously reads frames from serial connection with reconnect logic."""
        reconnect_cooldown = 2.0
        last_reconnect_attempt = 0.0

        while self._is_running:
            # Reconnect if not connected
            if not self.is_connected():
                now = time.time()
                if now - last_reconnect_attempt >= reconnect_cooldown:
                    last_reconnect_attempt = now
                    self._attempt_connection()
                time.sleep(0.5)
                continue

            try:
                if not self._serial_conn.is_open:
                    raise serial.SerialException("Serial port closed unexpectedly")

                line_bytes = self._serial_conn.readline()
                if not line_bytes:
                    time.sleep(0.01)
                    continue

                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                parsed = self._parse_telemetry_line(line_str)
                now_ts = time.time()

                if parsed is not None:
                    metrics, meta = parsed
                    self._consecutive_parse_errors = 0
                    self._total_frames_received += 1
                    self._chip_verified = meta["chip_verified"]
                    self._chip_id = meta["chip_id"]
                    self._last_reading_time = now_ts

                    # Determine human-readable state description
                    p = metrics["pitch"]
                    r = metrics["roll"]
                    if p <= config.HEAD_NOD_PITCH_THRESHOLD:
                        desc = f"Head Nodding Downward (Pitch {p}°)"
                    elif abs(r) >= config.HEAD_TILT_ROLL_THRESHOLD:
                        desc = f"Lateral Head Slump (Roll {r}°)"
                    else:
                        desc = "Normal Upright Driving Posture"

                    with self._lock:
                        self._latest_reading = SensorReading(
                            pitch=metrics["pitch"],
                            roll=metrics["roll"],
                            yaw=metrics["yaw"],
                            ax=metrics["ax"],
                            ay=metrics["ay"],
                            az=metrics["az"],
                            gx=metrics["gx"],
                            gy=metrics["gy"],
                            gz=metrics["gz"],
                            timestamp=now_ts,
                            is_valid=True,
                            is_connected=True,
                            mode="REAL",
                            port=self.port,
                            chip_verified=meta["chip_verified"],
                            chip_id=meta["chip_id"],
                            state_description=desc,
                            error_message="",
                            data_age_sec=0.0
                        )
                else:
                    self._consecutive_parse_errors += 1

            except (serial.SerialException, OSError) as e:
                # Handle USB disconnection or read error
                print(f"[ESP32 Bridge Warning] Hardware communication lost on {self.port}: {e}")
                with self._lock:
                    self._connected = False
                    self._connection_error = f"USB Disconnected: {e}"
                    if self._serial_conn:
                        try:
                            self._serial_conn.close()
                        except Exception:
                            pass
                        self._serial_conn = None

                    # Reset reading to invalid (never generate fake values)
                    self._latest_reading.is_valid = False
                    self._latest_reading.is_connected = False
                    self._latest_reading.state_description = "Hardware Link Disconnected"
                    self._latest_reading.error_message = str(e)
                time.sleep(1.0)
            except Exception as e:
                time.sleep(0.1)


# Standalone CLI validation test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIoT ESP32 Serial Telemetry Bridge CLI Monitor")
    parser.add_argument("--port", type=str, default=config.SERIAL_PORT, help="Serial COM port (default COM9)")
    parser.add_argument("--baud", type=int, default=config.SERIAL_BAUDRATE, help="Baud rate (default 115200)")
    args = parser.parse_args()

    print("=" * 70)
    print("  AIoT ESP32 HARDWARE SERIAL MONITOR")
    print("=" * 70)
    print(f"Target Port: {args.port} | Baud Rate: {args.baud}")
    print("Available Ports on host:")
    for p in ESP32SerialBridge.list_available_ports():
        print(f"  - {p['port']}: {p['description']}")
    print("-" * 70)

    bridge = ESP32SerialBridge(port=args.port, baudrate=args.baud)
    try:
        print("Polling telemetry (Press Ctrl+C to terminate)...")
        while True:
            reading = bridge.get_latest_reading()
            status = "VALID" if reading.is_valid else "WAITING/DISCONNECTED"
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Connected: {reading.is_connected} | "
                  f"Chip: {reading.chip_id} (Verified: {reading.chip_verified}) | "
                  f"Pitch: {reading.pitch}° Roll: {reading.roll}° | "
                  f"Raw Accel: ({reading.ax}, {reading.ay}, {reading.az}) | Age: {reading.data_age_sec}s")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down ESP32 bridge...")
        bridge.stop()
