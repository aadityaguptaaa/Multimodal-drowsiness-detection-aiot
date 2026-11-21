"""
Simulated MPU6050 IMU Sensor Module
Provides realistic head-pose dynamics for demonstration when physical hardware is not attached.
All readings are strictly labeled as SIMULATED.
"""

import time
import math
import random
from .base import IMUSensorBase, SensorReading

class SimulatedMPU6050(IMUSensorBase):
    """
    Produces clearly-labeled simulated pitch and roll readings.
    Supports controllable driving scenarios:
    - NORMAL: subtle driving posture vibration (pitch -2° to +2°, roll -3° to +3°)
    - HEAD_NOD: gradual or sudden downward nodding (pitch -25° to -35°) characteristic of microsleep
    - HEAD_TILT: driver head slumping to shoulder (roll ±28°)
    - SENSOR_DISCONNECTED: sensor communication failure simulation (is_valid=False)
    """
    def __init__(self, initial_state="NORMAL"):
        self.state = initial_state
        self.start_time = time.time()
        self.connected = True
        self.custom_pitch = None
        self.custom_roll = None

    def set_state(self, state_name):
        valid_states = ["NORMAL", "HEAD_NOD", "HEAD_TILT", "SENSOR_DISCONNECTED", "CUSTOM"]
        if state_name.upper() in valid_states:
            self.state = state_name.upper()
            if self.state == "SENSOR_DISCONNECTED":
                self.connected = False
            else:
                self.connected = True
        else:
            raise ValueError(f"Unknown state '{state_name}'. Choose from: {valid_states}")

    def set_custom_angles(self, pitch, roll):
        self.state = "CUSTOM"
        self.custom_pitch = float(pitch)
        self.custom_roll = float(roll)
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def get_mode(self) -> str:
        return "SIMULATED"

    def read(self) -> SensorReading:
        now = time.time()
        t = now - self.start_time

        if not self.connected or self.state == "SENSOR_DISCONNECTED":
            return SensorReading(
                pitch=0.0,
                roll=0.0,
                yaw=0.0,
                ax=0.0,
                ay=0.0,
                az=0.0,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                timestamp=now,
                is_valid=False,
                is_connected=False,
                mode="SIMULATED",
                port="N/A",
                chip_verified=False,
                chip_id="DISCONNECTED",
                state_description="Simulated Channel Inactive",
                error_message="Simulation disconnected state requested"
            )

        if self.state == "CUSTOM":
            pitch = self.custom_pitch if self.custom_pitch is not None else 0.0
            roll = self.custom_roll if self.custom_roll is not None else 0.0
            desc = "Controlled Test Static Angle"
        elif self.state == "HEAD_NOD":
            # Oscillates into deep forward head nod (-26° to -34°)
            base = -28.0 + 4.0 * math.sin(t * 1.5)
            pitch = round(base + random.uniform(-1.0, 1.0), 1)
            roll = round(random.uniform(-3.0, 3.0), 1)
            desc = "Simulated Abnormal Pitch: Head Nodding"
        elif self.state == "HEAD_TILT":
            # Slump to the right or left
            pitch = round(random.uniform(-5.0, 0.0), 1)
            base_roll = 28.0 + 3.0 * math.sin(t * 1.2)
            roll = round(base_roll + random.uniform(-1.0, 1.0), 1)
            desc = "Simulated Abnormal Roll: Lateral Head Slump"
        else:  # NORMAL
            # Subtle natural micro-movements
            pitch = round(1.5 * math.sin(t * 0.8) + random.uniform(-0.8, 0.8), 1)
            roll = round(2.0 * math.cos(t * 0.6) + random.uniform(-0.8, 0.8), 1)
            desc = "Simulated Normal Upright Driving Posture"

        yaw = round(3.0 * math.sin(t * 0.3) + random.uniform(-0.5, 0.5), 1)

        # Approximate raw accelerometer components from pitch & roll (g ~ 9.8 m/s²)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        ax = round(-math.sin(pitch_rad) * 9.806 + random.uniform(-0.1, 0.1), 2)
        ay = round(math.sin(roll_rad) * 9.806 + random.uniform(-0.1, 0.1), 2)
        az = round(math.cos(pitch_rad) * math.cos(roll_rad) * 9.806 + random.uniform(-0.1, 0.1), 2)

        # Gyro angular rate approximations (°/s)
        gx = round(random.uniform(-1.5, 1.5), 1)
        gy = round(random.uniform(-1.5, 1.5), 1)
        gz = round(random.uniform(-0.8, 0.8), 1)

        return SensorReading(
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            timestamp=now,
            is_valid=True,
            is_connected=True,
            mode="SIMULATED",
            port="TEST_FIXTURE",
            chip_verified=False,
            chip_id="SIMULATED_TEST_FIXTURE",
            state_description=desc,
            error_message="",
            data_age_sec=0.0
        )
