"""
Abstract Base Interface for Inertial Measurement Unit (IMU) Sensors
Allows seamless swapping between Simulated MPU6050 and Future Real Hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import time

@dataclass
class SensorReading:
    # Derived Head Angles (in degrees)
    pitch: float = 0.0          # In degrees: positive is head up, negative is head down (nodding)
    roll: float = 0.0           # In degrees: positive is tilt right, negative is tilt left
    yaw: float = 0.0            # In degrees: head turning left/right

    # Raw Accelerometer Measurements (m/s² or g)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # Raw Gyroscope Measurements (°/s)
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0

    # Telemetry Metadata
    timestamp: float = 0.0      # Unix epoch seconds
    is_valid: bool = False      # True if data is trustworthy, fresh, and within valid ranges
    is_connected: bool = False  # True if physical/simulated link is connected
    mode: str = "REAL"          # "REAL" or "SIMULATED"
    port: str = "COM9"          # Active or configured port
    chip_verified: bool = False # True only if hardware WHO_AM_I confirmed (e.g. 0x68)
    chip_id: str = "UNKNOWN"    # e.g. "MPU-6050 (0x68)" or "UNVERIFIED"
    state_description: str = "Sensor Offline" # e.g. "Normal Posture", "Hardware Disconnected"
    error_message: str = ""     # Description of error or disconnection reason
    data_age_sec: float = 0.0   # Elapsed seconds since this sample was produced

    def to_dict(self):
        return asdict(self)

class IMUSensorBase(ABC):
    """
    Common contract for IMU hardware and simulators.
    """
    @abstractmethod
    def read(self) -> SensorReading:
        """Returns the latest sensor reading."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the sensor connection is active."""
        pass

    @abstractmethod
    def get_mode(self) -> str:
        """Returns 'SIMULATED' or 'REAL'."""
        pass
