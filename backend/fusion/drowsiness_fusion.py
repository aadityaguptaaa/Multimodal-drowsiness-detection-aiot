"""
Modular Sensor Fusion Engine for AIoT Driver Drowsiness Detection
Transparent, configurable, rule-based multi-signal fusion layer.
Clearly identified as an experimental engineering fusion architecture.

Consumes:
1. Real-time eye-state prediction (Open / Closed, confidence)
2. MobileNetV2 face-level CNN classification
3. Rolling PERCLOS (10-second duration ratio) & Sustained Closure duration (Microsleep)
4. Blink frequency & eyelid dynamics
5. Physical ESP32 MPU-6050 Accelerometer & Gyroscope telemetry (when connected and fresh)

Graceful Degradation:
- Webcam-derived eye closure is primary drowsiness signal.
- Missing hardware NEVER disables working webcam inference.
- Missing camera data is NEVER fabricated.
- Separate statuses exposed for Camera, Face, Eyes, Model, PERCLOS, ESP32, and Motion Sensor.
"""

import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


class DrowsinessFusionEngine:
    """
    Combines independent visual and physical telemetry channels.
    Provides explainable status (NORMAL, WARNING, CRITICAL), risk score,
    and granular subsystem readiness indicators.
    """
    def __init__(
        self,
        cnn_threshold: float = config.CNN_DROWSY_THRESHOLD,
        perclos_drowsy_th: float = config.PERCLOS_DROWSY_THRESHOLD,
        perclos_warn_th: float = config.PERCLOS_WARNING_THRESHOLD,
        head_nod_pitch_th: float = config.HEAD_NOD_PITCH_THRESHOLD,
        head_tilt_roll_th: float = config.HEAD_TILT_ROLL_THRESHOLD,
        sustained_closure_th: float = config.SUSTAINED_CLOSURE_DROWSY_THRESHOLD_SEC
    ):
        self.cnn_threshold = cnn_threshold
        self.perclos_drowsy_th = perclos_drowsy_th
        self.perclos_warn_th = perclos_warn_th
        self.head_nod_pitch_th = head_nod_pitch_th
        self.head_tilt_roll_th = head_tilt_roll_th
        self.sustained_closure_th = sustained_closure_th

    def evaluate_readiness(
        self,
        camera_operational: bool,
        face_detected: bool,
        eyes_detected: bool,
        eye_model_loaded: bool,
        perclos_valid: bool,
        esp32_connected: bool,
        sensor_valid: bool,
        sensor_verified: bool
    ) -> Dict[str, Any]:
        """
        Computes separate readiness statuses for each subsystem.
        """
        motion_operational = esp32_connected and sensor_valid and sensor_verified

        if not camera_operational:
            overall_readiness = "NOT_READY_NO_CAMERA"
        elif not face_detected:
            overall_readiness = "DEGRADED_NO_FACE"
        elif not esp32_connected:
            overall_readiness = "ACTIVE_VISION_ONLY"
        elif not motion_operational:
            overall_readiness = "ACTIVE_UNVERIFIED_IMU"
        else:
            overall_readiness = "FULL_MULTIMODAL_ACTIVE"

        return {
            "camera_operational": bool(camera_operational),
            "face_detected": bool(face_detected),
            "eyes_detected": bool(eyes_detected),
            "eye_model_loaded": bool(eye_model_loaded),
            "perclos_valid": bool(perclos_valid),
            "esp32_connected": bool(esp32_connected),
            "motion_sensor_operational": bool(motion_operational),
            "overall_detection_readiness": overall_readiness,
            "fusion_methodology": "Rule-Based Multi-Signal Synchronization (Experimental)"
        }

    def fuse(
        self,
        cnn_result: Dict[str, Any],
        perclos_result: Dict[str, Any],
        blink_result: Dict[str, Any],
        sensor_reading: Any,
        eye_state_result: Optional[Dict[str, Any]] = None,
        camera_operational: bool = True,
        face_detected: bool = True,
        eyes_detected: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesizes visual and physical telemetry into an explainable decision.
        """
        contributing_factors = []
        critical_cues = 0
        warning_cues = 0

        # Convert sensor reading to dict if needed
        sensor_dict = sensor_reading.to_dict() if hasattr(sensor_reading, "to_dict") else dict(sensor_reading)

        # 0. Subsystem Readiness Evaluation
        eye_model_loaded = (eye_state_result is not None and eye_state_result.get("is_valid", False))
        esp32_connected = bool(sensor_dict.get("is_connected", False))
        sensor_valid = bool(sensor_dict.get("is_valid", False))
        sensor_verified = bool(sensor_dict.get("chip_verified", False))
        perclos_valid = bool(perclos_result.get("valid_observation", False))

        readiness = self.evaluate_readiness(
            camera_operational=camera_operational,
            face_detected=face_detected,
            eyes_detected=eyes_detected,
            eye_model_loaded=eye_model_loaded,
            perclos_valid=perclos_valid,
            esp32_connected=esp32_connected,
            sensor_valid=sensor_valid,
            sensor_verified=sensor_verified
        )

        # 1. Evaluate Primary Signal: Sustained Eye Closure & Microsleep
        sustained_closure_sec = perclos_result.get("sustained_closure_sec", 0.0)
        is_sustained_closure = perclos_result.get("is_sustained_closure", False) or (sustained_closure_sec >= self.sustained_closure_th)

        if is_sustained_closure:
            critical_cues += 2  # Immediate priority
            contributing_factors.append(
                f"Microsleep Watchdog: Eyes continuously closed for {sustained_closure_sec:.1f}s (Threshold: {self.sustained_closure_th}s)"
            )

        # 2. Evaluate Trained Eye-State Model Inference
        eye_score = 0.0
        if eye_state_result and eye_state_result.get("is_valid", False):
            eye_state = eye_state_result.get("state", "open")
            eye_conf = eye_state_result.get("confidence", 0.0)
            closed_prob = eye_state_result.get("closed_probability", 0.0)
            eye_score = closed_prob * 100.0

            if eye_state == "closed" and eye_conf >= 0.70:
                warning_cues += 1
                contributing_factors.append(f"Eye Model: High confidence eye closure detected ({int(eye_conf * 100)}%)")
        else:
            # Model not ready or face missing
            pass

        # 3. Evaluate Rolling PERCLOS Duration
        perclos_score = 0.0
        if perclos_valid:
            p_ratio = perclos_result.get("perclos_ratio", 0.0)
            p_percent = perclos_result.get("perclos_percent", 0.0)

            if p_ratio >= self.perclos_drowsy_th:
                critical_cues += 1
                contributing_factors.append(
                    f"PERCLOS Duration: {p_percent}% closure in {perclos_result.get('window_duration_sec', 10)}s window (Threshold: {int(self.perclos_drowsy_th * 100)}%)"
                )
                perclos_score = 75.0 + min(25.0, ((p_ratio - self.perclos_drowsy_th) / max(0.01, 1.0 - self.perclos_drowsy_th)) * 25.0)
            elif p_ratio >= self.perclos_warn_th:
                warning_cues += 1
                contributing_factors.append(f"PERCLOS Duration: Elevated closure trend ({p_percent}%)")
                perclos_score = 50.0 + min(24.0, ((p_ratio - self.perclos_warn_th) / max(0.01, self.perclos_drowsy_th - self.perclos_warn_th)) * 24.0)
            else:
                perclos_score = (p_ratio / max(0.01, self.perclos_warn_th)) * 40.0

        # 4. Evaluate MobileNetV2 Face Model
        cnn_score = 0.0
        cnn_valid = cnn_result.get("is_valid", False) and cnn_result.get("model_loaded", False)
        if cnn_valid:
            prob = cnn_result.get("smoothed_probability", cnn_result.get("probability", 0.0))
            cnn_score = prob * 100.0
            if prob >= 0.78:
                critical_cues += 1
                contributing_factors.append(f"MobileNetV2 CNN: Drowsy facial expression ({int(prob * 100)}% confidence)")
            elif prob >= self.cnn_threshold:
                warning_cues += 1
                contributing_factors.append(f"MobileNetV2 CNN: Moderate facial fatigue signs ({int(prob * 100)}%)")

        # 5. Evaluate Blink Dynamics
        blink_score = 0.0
        if blink_result:
            avg_dur = blink_result.get("average_duration_ms", 200.0)
            trend = blink_result.get("blink_trend", "NORMAL")
            bpm = blink_result.get("blinks_per_minute", 15.0)

            if "SLUGGISH" in trend or avg_dur >= 380.0:
                warning_cues += 1
                contributing_factors.append(f"Blink Dynamics: Slow eye reopening ({avg_dur} ms average duration)")
                blink_score = 60.0
            elif "FLUTTERING" in trend:
                warning_cues += 1
                contributing_factors.append(f"Blink Dynamics: Rapid eyelid fluttering ({bpm} BPM)")
                blink_score = 50.0

        # 6. Evaluate Secondary Physical Signal: ESP32 MPU-6050
        # Physical motion is a corroborating signal; an unmounted IMU resting on a desk
        # must NOT trigger driver alert if the driver's eyes are open and alert.
        head_score = 0.0
        if esp32_connected and sensor_valid:
            pitch = sensor_dict.get("pitch", 0.0)
            roll = sensor_dict.get("roll", 0.0)

            has_visual_fatigue = (
                is_sustained_closure or
                (perclos_valid and perclos_result.get("perclos_ratio", 0.0) >= self.perclos_warn_th) or
                (cnn_valid and cnn_result.get("smoothed_probability", 0.0) >= self.cnn_threshold) or
                (eye_state_result and eye_state_result.get("state") == "closed")
            )

            if pitch <= self.head_nod_pitch_th:
                head_score = 85.0
                if has_visual_fatigue:
                    critical_cues += 1
                    contributing_factors.append(f"ESP32 IMU: Head nodding downward corroborated by visual fatigue (Pitch: {pitch:.1f}°)")
                else:
                    warning_cues += 1
                    contributing_factors.append(f"ESP32 IMU: Head downward angle detected (Pitch: {pitch:.1f}°)")
            elif abs(roll) >= self.head_tilt_roll_th:
                head_score = 65.0
                if has_visual_fatigue:
                    warning_cues += 1
                    contributing_factors.append(f"ESP32 IMU: Lateral head slump corroborated by fatigue (Roll: {roll:.1f}°)")
                # If eyes are wide open and no fatigue, do not penalize a desk-resting sensor
        else:
            if not esp32_connected:
                contributing_factors.append("Note: ESP32 hardware bridge offline; monitoring solely via optical vision")

        # 7. Decision Rule Hierarchy (Safety Critical)
        # Primary ground truth: eyes and PERCLOS
        if is_sustained_closure:
            overall_state = "CRITICAL"  # Microsleep watchdog always overrides
        elif perclos_valid and perclos_result.get("perclos_ratio", 0.0) >= self.perclos_drowsy_th:
            overall_state = "CRITICAL"
        elif critical_cues >= 2:
            overall_state = "CRITICAL"
        elif critical_cues >= 1 and warning_cues >= 1:
            overall_state = "CRITICAL"
        elif critical_cues >= 1 or warning_cues >= 1:
            overall_state = "WARNING"
        else:
            overall_state = "NORMAL"

        # Calculate Normalized Composite Risk Index [0, 100]
        # Active weights re-normalize if hardware is offline
        w_cnn = config.FUSION_WEIGHT_CNN
        w_perclos = config.FUSION_WEIGHT_PERCLOS
        w_blink = config.FUSION_WEIGHT_BLINK
        w_head = config.FUSION_WEIGHT_HEAD if (esp32_connected and sensor_valid) else 0.0

        total_weight = w_cnn + w_perclos + w_blink + w_head
        if total_weight > 0:
            norm_cnn = w_cnn / total_weight
            norm_perclos = w_perclos / total_weight
            norm_blink = w_blink / total_weight
            norm_head = w_head / total_weight
        else:
            norm_cnn, norm_perclos, norm_blink, norm_head = 0.4, 0.4, 0.2, 0.0

        composite_score = (
            (cnn_score * norm_cnn) +
            (perclos_score * norm_perclos) +
            (blink_score * norm_blink) +
            (head_score * norm_head)
        )
        if is_sustained_closure:
            composite_score = max(composite_score, 90.0)

        composite_score = round(min(100.0, max(0.0, composite_score)), 1)

        # Confidence of overall assessment
        active_sources = sum([
            camera_operational and face_detected,
            perclos_valid,
            cnn_valid,
            esp32_connected and sensor_valid
        ])
        if active_sources >= 3:
            confidence = "HIGH"
        elif active_sources == 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if not contributing_factors:
            contributing_factors.append("All primary visual and inertial telemetry channels nominal")

        return {
            "overall_state": overall_state,
            "composite_score": composite_score,
            "contributing_factors": contributing_factors,
            "confidence_level": confidence,
            "readiness": readiness,
            "signals": {
                "eye_model": eye_state_result or {"is_valid": False, "state": "MODEL_NOT_READY"},
                "cnn": cnn_result,
                "perclos": perclos_result,
                "blink": blink_result,
                "sensor": sensor_dict
            }
        }
