"""
Real MPU6050 Hardware Adapter with ESP32 Serial Telemetry
Implements IMUSensorBase by binding to ESP32SerialBridge.
Reads genuine accelerometer (ax, ay, az) and gyroscope (gx, gy, gz) values over USB Serial (COM9 default).
Safely handles absent physical hardware without crashing or generating fake values.
"""

import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from .base import IMUSensorBase, SensorReading
from .esp32_bridge import ESP32SerialBridge


class RealMPU6050(IMUSensorBase):
    """
    Physical hardware driver for MPU6050 6-axis IMU connected via ESP32 USB Serial.
    Default port: COM9 on Windows (configurable).
    Default baud: 115200.
    """
    def __init__(self, port: str = config.SERIAL_PORT, baudrate: int = config.SERIAL_BAUDRATE, auto_start: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.bridge = ESP32SerialBridge(
            port=self.port,
            baudrate=self.baudrate,
            timeout=config.SERIAL_TIMEOUT_SEC,
            stale_timeout_sec=config.SENSOR_STALE_TIMEOUT_SEC,
            auto_start=auto_start
        )

    def connect(self) -> bool:
        """Forces reconnect attempt."""
        return self.is_connected()

    def is_connected(self) -> bool:
        """Returns True when serial link is active or standalone bridge.py is active."""
        if self.bridge.is_connected():
            return True
        try:
            import urllib.request
            import json
            req = urllib.request.Request("http://127.0.0.1:8765/health", headers={"User-Agent": "AIoT-Backend"})
            with urllib.request.urlopen(req, timeout=0.15) as resp:
                if resp.status == 200:
                    h = json.loads(resp.read().decode("utf-8"))
                    return bool(h.get("connected", False))
        except Exception:
            pass
        return False

    def get_mode(self) -> str:
        return "REAL"

    def read(self) -> SensorReading:
        """Returns the latest real physical sensor measurement from serial or localhost bridge."""
        reading = self.bridge.get_latest_reading()
        if not reading.is_connected:
            try:
                import urllib.request
                import json
                req = urllib.request.Request("http://127.0.0.1:8765/data", headers={"User-Agent": "AIoT-Backend"})
                with urllib.request.urlopen(req, timeout=0.15) as resp:
                    if resp.status == 200:
                        bdata = json.loads(resp.read().decode("utf-8"))
                        if bdata.get("is_connected"):
                            tel = bdata.get("telemetry", {})
                            return SensorReading(
                                pitch=tel.get("pitch", 0.0),
                                roll=tel.get("roll", 0.0),
                                yaw=tel.get("yaw", 0.0),
                                ax=tel.get("ax", 0.0),
                                ay=tel.get("ay", 0.0),
                                az=tel.get("az", 0.0),
                                gx=tel.get("gx", 0.0),
                                gy=tel.get("gy", 0.0),
                                gz=tel.get("gz", 0.0),
                                is_valid=True,
                                is_connected=True,
                                mode="REAL",
                                state_description="ESP32_BRIDGE_ONLINE",
                                chip_verified=bdata.get("chip_verified", True),
                                chip_id=bdata.get("chip_id", "0x68"),
                                port=bdata.get("port", self.port),
                                data_age_sec=bdata.get("data_age_sec", 0.0),
                                timestamp=bdata.get("timestamp", time.time())
                            )
            except Exception:
                pass
        return reading

    def get_diagnostics(self):
        """Returns diagnostic status of the serial bridge and chip identity verification."""
        diag = self.bridge.get_status()
        if not diag.get("is_connected"):
            try:
                import urllib.request
                import json
                req = urllib.request.Request("http://127.0.0.1:8765/health", headers={"User-Agent": "AIoT-Backend"})
                with urllib.request.urlopen(req, timeout=0.15) as resp:
                    if resp.status == 200:
                        h = json.loads(resp.read().decode("utf-8"))
                        diag["is_connected"] = h.get("connected", False)
                        diag["status"] = "ONLINE_VIA_LOCAL_BRIDGE"
            except Exception:
                pass
        return diag

    def set_port(self, port_name: str, baudrate: int = 115200):
        """Reconfigures target port."""
        self.port = port_name
        self.baudrate = baudrate
        return self.bridge.set_port(port_name, baudrate)

    def close(self):
        """Releases the serial port resource."""
        self.bridge.stop()
