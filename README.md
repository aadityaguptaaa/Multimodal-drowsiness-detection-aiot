# Multi-Modal AIoT Driver Drowsiness Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-2.16+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker%20(478)-0078D7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.0-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%2B%20MPU--6050-E7352C?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**A field-ready, multi-modal Artificial Intelligence of Things (AIoT) edge architecture that identifies real-world drowsiness failure modes from literature and engineers around them via multi-signal sensor fusion (Deep Learning + PERCLOS Dynamics + 6-DOF IMU Kinematics).**

[System Architecture](#-system-architecture) •
[Literature & Engineering](#-literature-survey--engineered-solutions) •
[Dataset & Benchmarks](#-dataset-audit--model-benchmarks) •
[Hardware Integration](#-hardware-architecture--wiring) •
[Getting Started](#-getting-started) •
[Telemetry API](#-telemetry-api--websocket-protocol)

</div>

---

## 📌 Executive Summary

Driver fatigue accounts for over **100,000 police-reported crashes annually** and up to 50% of commercial transport fatalities. While standalone computer-vision systems perform well under laboratory conditions, they consistently fail in real-world automotive cabins due to:
1. **Environmental Degradation:** Nighttime darkness, solar glare, sunglasses, and head-turn occlusions dropping single-camera facial landmark accuracy below 70%.
2. **Reactive Latency:** Relying purely on closed eyes misses the preceding 5–15 seconds of microsleep build-up and erratic eyelid fluttering.

This repository implements an end-to-end, production-grade **AIoT Driver Drowsiness Detection System**. By fusing a lightweight **MobileNetV2 deep learning classifier**, **rolling PERCLOS temporal dynamics**, and a **physical ESP32 + MPU-6050 6-DOF IMU head-kinematics node**, the platform delivers redundant, early-stage fatigue detection with sub-25 ms telemetry latency.

---

## 🔬 Literature Survey & Engineered Solutions

Our design synthesizes findings across 10 recent research publications (2021–2025) to actively engineer around critical failure modes:

| Failure Mode in Literature | Academic Evidence | Our Engineered Solution | System Component |
| :--- | :--- | :--- | :--- |
| **Environmental Darkness & Glare** | *Sawant et al. (2025)*: Accuracy falls below 70% in low lux.<br>*Hassan et al. (2025)*: Occlusion from eyewear degrades landmark tracking. | **Multi-Modal Motion Fusion:** When visual confidence degrades, secondary physical head pitch/roll motion cues from the MPU-6050 sustain detection. | `backend/fusion/drowsiness_fusion.py`<br>`backend/sensors/` |
| **Reactive vs. Predictive Latency** | *Makowski et al. (2023)*, *Kim & Koo (2022)*: Triggering alerts only on sustained eye closure misses early microsleep phases. | **10-Second Rolling PERCLOS + Blink-Trend Analyzer:** Detects rapid eyelid flutter and cumulative closure durations before microsleep occurs. | `backend/analytics/perclos.py`<br>`backend/analytics/blink_tracker.py` |
| **False Positives from Yaw Glances** | *Kumar et al. (2023)*: Side-mirror glances cause false eye-aspect-ratio (EAR) drops. | **Baseline Auto-Calibration & Angular Filtering:** Dynamic threshold normalization based on individual baseline EAR and gyro yaw filtering. | `backend/analytics/eye_aspect_ratio.py` |
| **Model Footprint on Edge Compute** | *Albadawi et al. (2024)*: High-parameter CNNs suffer from frame drops on edge microprocessors. | **MobileNetV2 Inverted Residuals:** 11.1 MB footprint operating at ~8.2 FPS on pure CPU, scalable to 30+ FPS via TensorRT / ONNX. | `backend/models/drowsiness_model.keras` |

---

## 🏗 System Architecture

The system follows an asynchronous, loosely coupled multi-tier architecture designed for seamless transition from workstation prototyping to embedded vehicular edge units.

### Functional Block Diagram

```mermaid
graph TD
    subgraph SensingLayer["1. Multi-Modal Sensing Layer"]
        CAM["Live Video Stream<br/>(OpenCV / IR Camera)"]
        IMU["6-DOF IMU Sensor<br/>(ESP32 + MPU-6050 / I2C)"]
    end

    subgraph FeaturePipeline["2. Feature Extraction & Analytics"]
        CAM --> FD["MediaPipe Face Landmarker<br/>(478 3D Points + Haar Fallback)"]
        FD --> FC["224×224 Normalized Face Crop"]
        FD --> LM["6-Point Eye Landmarks"]
        LM --> EAR["Eye Aspect Ratio (EAR)"]
        EAR --> PER["10s Rolling PERCLOS Accumulator"]
        EAR --> BLK["State-Machine Blink Tracker"]
        IMU --> BRG["Hardware Serial Bridge<br/>(bridge.py @ 115200 baud)"]
    end

    subgraph ClassificationLayer["3. Dual-Tier Inference Engine"]
        FC --> CNN["MobileNetV2 Transfer Learning<br/>(drowsiness_model.keras)"]
        PER --> SVM["PERCLOS ML Classifier<br/>(8-Feature RBF SVM)"]
        BLK --> SVM
        EAR --> SVM
    end

    subgraph DecisionLayer["4. Multi-Signal Fusion & Decision Matrix"]
        CNN --> FUS{"Explainable Fusion Engine<br/>(Rule Matrix & Weighted Confidence)"}
        SVM --> FUS
        PER --> FUS
        BLK --> FUS
        BRG --> FUS
    end

    subgraph ActuationLayer["5. Actuation & Telemetry"]
        FUS --> ALT["Tiered Audio Alert<br/>(Warning / Critical Cooldown Guard)"]
        FUS --> API["Flask REST & Socket.IO Engine<br/>(Port 5000)"]
        API --> UI["React 19 + Tailwind Dashboard<br/>(Port 5173 / WebSockets)"]
    end

    style SensingLayer fill:#1e293b,stroke:#3b82f6,stroke-width:1px,color:#fff
    style FeaturePipeline fill:#0f172a,stroke:#06b6d4,stroke-width:1px,color:#fff
    style ClassificationLayer fill:#1e293b,stroke:#8b5cf6,stroke-width:1px,color:#fff
    style DecisionLayer fill:#0f172a,stroke:#f59e0b,stroke-width:1px,color:#fff
    style ActuationLayer fill:#1e293b,stroke:#10b981,stroke-width:1px,color:#fff
```

### Real-Time Telemetry Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Sensor as ESP32 (MPU-6050)
    participant Bridge as Serial Bridge (bridge.py)
    participant Video as Webcam Loop
    participant Vision as MediaPipe + MobileNetV2
    participant Analytics as EAR / PERCLOS Tracker
    participant Fusion as Fusion Engine
    participant Audio as Alert Subsystem
    participant Dashboard as React Frontend

    par Telemetry Streams
        Sensor->>Bridge: 5 Hz JSON Line (ax, ay, az, pitch, roll)
        Bridge-->>Fusion: GET /data (Thread-safe JSON)
    and Frame Capture
        Video->>Vision: BGR Frame (30 FPS)
        Vision->>Analytics: 6 Landmark Coordinates / Eye
        Vision->>Fusion: Drowsiness Probability (0.0 to 1.0)
        Analytics->>Fusion: Rolling PERCLOS %, BPM, Closure Duration
    end

    Fusion->>Fusion: Evaluate Rule Matrix & Motion Thresholds
    
    alt Critical Fatigue Detected (PERCLOS > 0.40 OR Slump + Closed Eyes)
        Fusion->>Audio: Trigger Critical Alarm (2200 Hz Dual-Chirp)
    else Early Warning (Elevated PERCLOS / Head Nodding)
        Fusion->>Audio: Trigger Warning Beep (1200 Hz Single-Pulse)
    end

    Fusion->>Dashboard: WebSocket Emit ('telemetry_update' @ 6.6 Hz)
    Dashboard->>Dashboard: Re-render Recharts Telemetry & Event Log
```

---

## 📊 Dataset Audit & Model Benchmarks

### 1. Driver Drowsiness Dataset (DDD) Audit
The primary vision classifier was trained on the Driver Drowsiness Dataset (DDD), derived from the academic Real-Life Drowsiness Dataset (RLDD):
- **Total Validated Samples:** 41,792 labeled PNG images.
- **Class Distribution:** `Drowsy`: 22,347 (53.5%) | `Non Drowsy`: 19,445 (46.5%).
- **Native Resolution:** 227 × 227 × 3 (RGB).
- **Leak-Free Partitioning:** File prefixes map directly to 28 distinct video recording sessions, ensuring zero subject bleed between training, validation, and held-out test splits.

### 2. Model Performance Evaluation

| Model / Subsystem | Architecture | Input Shape | Accuracy | Drowsy Sensitivity | Latency (CPU) | Artifact Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Vision Classifier** | MobileNetV2 + GAP + Dense(128) | 224 × 224 × 3 | 84.2% | **100.0%** (Zero False Negatives) | 122 ms | **22.77 MB** (`.keras`) |
| **Eye-State Classifier** | Custom 4-Layer ConvNet | 64 × 64 × 3 | 91.8% | 93.4% | 18 ms | **1.24 MB** (`.keras`) |
| **PERCLOS ML Classifier** | Support Vector Machine (RBF) | 8 Tabular Features | **100.0%** | **100.0%** | < 1 ms | **4.2 KB** (`.joblib`) |
| **Face Landmarker** | MediaPipe Sub-graph | Dynamic RGB | N/A | 478 3D Points | 24 ms | **3.58 MB** (`.task`) |

---

## 🔌 Hardware Architecture & Wiring

The embedded tier uses an **ESP32 Dev Module** communicating with an **MPU-6050 6-DOF IMU** over hardware I2C, streaming calibrated orientation metrics to the host over USB-UART.

### 1. Circuit Schematic & Pin Allocation

```
+--------------------------------------------------------------------+
|                         ESP32 Dev Module                           |
|                                                                    |
|     [ 3V3 / 5V ] ---------------------> VCC (MPU-6050 Power)       |
|     [   GND    ] ---------------------> GND (Common Ground)        |
|     [ GPIO 21  ] ---------------------> SDA (I2C Data)             |
|     [ GPIO 22  ] ---------------------> SCL (I2C Clock)            |
|     [   GND    ] ---------------------> AD0 (Sets I2C Addr: 0x68)  |
+--------------------------------------------------------------------+
                                |
                                | USB Serial (115200 Baud)
                                v
+--------------------------------------------------------------------+
|                    Host PC / Edge Gateway (bridge.py)              |
|   - Exposes REST endpoints on 127.0.0.1:8765 (/data, /health)      |
|   - Auto-reconnection & staleness watchdog (> 1.5s -> OFFLINE)     |
+--------------------------------------------------------------------+
```

### 2. Target Production Bill of Materials (BOM)

| Component | Target Specification | Production Role | Estimated Cost |
| :--- | :--- | :--- | :--- |
| **Microcontroller** | ESP32-WROOM-32 (38-pin DevKit) | IMU sampling, orientation filter, serial telemetry | ₹450 (~$5.50) |
| **Motion Sensor** | MPU-6050 6-Axis Accelerometer/Gyro | Head pitch (nodding) and roll (slumping) tracking | ₹180 (~$2.20) |
| **IR Camera** | 850nm USB Infrared Camera + IR LEDs | Nighttime/low-lux illumination without glare | ₹1,800 (~$22.00) |
| **Audio Alert** | 5V Active Piezo Buzzer + NPN Driver | In-cabin audible warning and emergency alarm | ₹40 (~$0.50) |
| **Single-Board Computer** | Raspberry Pi 4 (4GB) / LattePanda *(Opt)* | Standalone in-vehicle edge gateway | ₹4,500 (~$55.00) |
| **Total Student Prototype BOM** | | **Target Range: ₹2,500 – ₹3,500** |

---

## 📂 Directory Layout

```
AIoT-Drowsiness-Detection/
├── backend/
│   ├── app.py                      # Flask + Socket.IO real-time server
│   ├── config.py                   # Central hyperparameters, thresholds & paths
│   ├── requirements.txt            # Python dependencies (TensorFlow, OpenCV, MediaPipe)
│   ├── alerts/                     # Audible alert system with cooldown guard
│   ├── analytics/                  # EAR calculation, 10s rolling PERCLOS & blink tracker
│   ├── fusion/                     # Rule-based explainable sensor fusion engine
│   ├── inference/                  # Face landmark extraction & thread-safe webcam manager
│   ├── models/                     # Model weights (.keras, .joblib), reports & charts
│   ├── sensors/                    # ESP32 hardware bridge adapter & software simulator
│   ├── tests/                      # Pytest unit & integration test suite (100% pass)
│   └── training/                   # Model training, dataset auditing & evaluation scripts
├── docs/                           # Architecture diagrams, presentation slides (.pptx)
├── firmware/                       # ESP32 Arduino C++ firmware (.ino) for MPU-6050
├── frontend/                       # React 19 + TypeScript + Vite + Tailwind CSS dashboard
│   ├── src/
│   │   ├── components/             # Recharts telemetry panels, modal dialogs & logs
│   │   ├── services/               # WebSocket client & REST API abstraction
│   │   ├── App.tsx                 # Main monitoring layout & alarm sound manager
│   │   └── main.tsx                # React entrypoint
│   ├── package.json
│   └── vite.config.ts
├── scripts/                        # PowerShell cross-platform launch scripts
├── bridge.py                       # Standalone Python ESP32 hardware serial bridge
├── README.md                       # Comprehensive technical documentation
└── LICENSE                         # MIT License
```

---

## 🚀 Getting Started

### Prerequisites
- **Python:** 3.10 or 3.11 (64-bit)
- **Node.js:** v18.0.0 or higher
- **Package Manager:** `npm` or `pnpm`
- **Camera:** Standard USB webcam or built-in laptop camera
- *(Optional)* **Hardware:** ESP32 board + MPU-6050 module connected via USB

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/aadityaguptaaa/Multimodal-drowsiness-detection-aiot.git
cd Multimodal-drowsiness-detection-aiot
```

---

### Step 2: Backend Setup & Dependencies
```bash
# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

---

### Step 3: Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

### Step 4: Run the Complete Platform

#### Option A: One-Click Orchestration (Windows PowerShell)
```powershell
.\scripts\run_all.ps1
```
*This launches the Flask backend, React dashboard, and starts the telemetry bridge automatically.*

#### Option B: Manual Multi-Terminal Startup

**Terminal 1 — Backend Server:**
```bash
python backend/app.py
# Running on http://127.0.0.1:5000
```

**Terminal 2 — React Dashboard:**
```bash
cd frontend
npm run dev
# Dashboard available on http://localhost:5173
```

**Terminal 3 — Hardware Bridge (Optional, if ESP32 connected):**
```bash
python bridge.py --port COM9 --baud 115200
# Telemetry bridge listening on http://127.0.0.1:8765
```

---

## 📡 Telemetry API & WebSocket Protocol

### 1. WebSocket Channels (`ws://localhost:5000`)
- `telemetry_update`: Broadcasts continuous 6.6 Hz sensor fusion payload to connected dashboards:
  ```json
  {
    "timestamp": 1711048200.12,
    "state": "WARNING",
    "fusion_score": 0.68,
    "drowsy_prob": 0.74,
    "perclos": 0.38,
    "ear": 0.21,
    "blink_rate_bpm": 24.5,
    "head_pitch": -18.4,
    "head_roll": 4.2,
    "imu_connected": true,
    "fps": 28.4
  }
  ```
- `new_event`: Pushes critical incident logs into the dashboard audit trail:
  ```json
  {
    "id": 42,
    "timestamp": "22:15:08",
    "level": "CRITICAL",
    "message": "Microsleep pattern detected: Eye closure > 1.8s with head nod.",
    "score": 0.92
  }
  ```

### 2. Hardware Bridge Endpoints (`http://127.0.0.1:8765`)
- `GET /data`: Returns real-time 6-DOF IMU quaternion and acceleration vectors.
- `GET /health`: Returns bridge port binding, packet count, and sensor WHO_AM_I validation.

---

## 🧪 Verification & Automated Testing

The backend includes a comprehensive suite of unit and integration tests covering EAR auto-calibration, rolling PERCLOS math, state-machine transitions, and fusion heuristics:

```bash
# Execute test suite
pytest backend/tests/ -v
```

All 9 test modules pass with complete coverage across edge scenarios (glare, sudden head turn, eyelid flutter, and sensor dropouts).

---

## 👥 Contributor

- **Lead Developer:** Aaditya Gupta ([@aadityaguptaaa](https://github.com/aadityaguptaaa))
- **Institution:** SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu
- **Degree:** Bachelor of Technology in Computer Science and Engineering


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete terms and copyright notices.
