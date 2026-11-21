# AIoT Driver Drowsiness Detection: Technical Architecture

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Sensing & Video Capture
        CAM["Laptop Webcam (OpenCV)<br/>[Future: USB IR Camera]"]
        IMU["SimulatedMPU6050 (Head Pitch/Roll)<br/>[Future: Physical I2C MPU6050]"]
    end

    subgraph Computer Vision & Feature Extraction
        CAM --> FD["UnifiedFaceDetector<br/>(MediaPipe 478 Landmarks / Haar Fallback)"]
        FD --> FC["224x224 Face Crop"]
        FD --> LM["Eye Landmarks (6 points/eye)"]
        FC --> CNN["MobileNetV2 Transfer Learning<br/>(drowsiness_model.keras)"]
        LM --> EAR["Eye Aspect Ratio (EAR)"]
        EAR --> PER["RollingPERCLOS (10s window)"]
        EAR --> BLK["BlinkTracker (BPM & Trend)"]
    end

    subgraph Machine Learning Classification
        PER --> PER_ML["PERCLOS ML Model<br/>(perclos_model.joblib - Kumar et al. 2023)"]
        BLK --> PER_ML
        EAR --> PER_ML
    end

    subgraph Sensor Fusion & Decision Engine
        CNN --> FUS["DrowsinessFusionEngine<br/>(Tiered Explainable Multi-Signal Fusion)"]
        PER --> FUS
        PER_ML --> FUS
        BLK --> FUS
        IMU --> FUS
    end

    subgraph Actuation & Telemetry
        FUS --> ALT["AudioAlertSystem<br/>(Windows Beep / Cooldown Guard)"]
        FUS --> API["Flask REST & Socket.IO Server<br/>(Port 5000)"]
        API --> UI["React + Vite + Tailwind Dashboard<br/>(Port 5173)"]
    end
```

---

## 2. Data Flow Pipeline

```mermaid
sequenceDiagram
    participant Cam as Webcam Frame
    participant Face as Face Detector
    participant CNN as MobileNetV2
    participant Analytics as EAR / PERCLOS / Blinks
    participant Sensor as MPU6050 IMU
    participant Fusion as Fusion Engine
    participant Alert as Audio Alert
    participant UI as React Dashboard

    Cam->>Face: Capture BGR Frame (30 FPS)
    Face->>CNN: Normalized Face Crop [-1, 1]
    Face->>Analytics: 6 Landmark Coordinates / Eye
    CNN-->>Fusion: Drowsiness Probability & Class
    Analytics-->>Fusion: Rolling PERCLOS %, BPM, Sluggish Trend
    Sensor-->>Fusion: Head Pitch (Nod) & Roll (Tilt)
    Fusion->>Fusion: Evaluate Rule Matrix & Composite Score
    alt Critical Drowsiness Confirmed
        Fusion->>Alert: Trigger Buzzer (Double Tone 2200Hz)
    else Early Fatigue Warning
        Fusion->>Alert: Trigger Warning (Single Tone 1200Hz)
    end
    Fusion->>UI: Emit Telemetry Packet over WebSockets (6.6 Hz)
```

---

## 3. Software vs. Future Hardware Interface Matrix

| Component | Current Software Prototype (Review Day) | Phase 2 Physical Hardware Integration | Software Adapter File |
| :--- | :--- | :--- | :--- |
| **Camera** | Built-in Laptop RGB Webcam | 850nm USB Infrared Camera with IR LEDs | `backend/inference/webcam.py` |
| **Motion Sensor** | `SimulatedMPU6050` (Normal, Nod, Tilt) | Physical MPU6050 on I2C bus (0x68) | `backend/sensors/real_mpu6050.py` |
| **Audio Buzzer** | Laptop Speaker (`winsound.Beep`) | 5V Piezo Buzzer on GPIO Transistor | `backend/alerts/audio_alert.py` |
| **Compute Board** | Windows 11 PC (Python 3.11) | Raspberry Pi 4 (4GB) or ESP32-S3 | `backend/config.py` |
| **Target Budget** | Zero additional hardware cost | **₹3,000 – ₹4,000** student budget | BOM in `docs/` |
