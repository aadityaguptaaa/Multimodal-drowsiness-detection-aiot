"""
Thread-Safe Live Webcam Capture & Inference Pipeline
Orchestrates:
- OpenCV VideoCapture with permission & error handling
- Face Detection & Landmark Extraction (UnifiedFaceDetector)
- Real Eye-State CNN Model Inference (EyeStatePredictor)
- Eye Openness & EAR Calculation (EyeTracker)
- 10-Second Rolling PERCLOS with Gap & Microsleep Protection (RollingPERCLOS)
- Blink Detection & Frequency/Trend Tracking (BlinkTracker)
- Deep Learning CNN Inference with Temporal Smoothing
- Physical ESP32 Hardware Integration (RealMPU6050 via ESP32SerialBridge)
- Explainable Multi-Signal Fusion (DrowsinessFusionEngine)
- Audio Alert Triggering with Cooldown Guard
"""

import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from inference.preprocessing import preprocess_image
from inference.face_detector import UnifiedFaceDetector
from inference.predict_eye import EyeStatePredictor
from analytics.eye_aspect_ratio import EyeTracker
from analytics.perclos import RollingPERCLOS
from analytics.blink_tracker import BlinkTracker
from sensors.simulated_mpu6050 import SimulatedMPU6050
from sensors.real_mpu6050 import RealMPU6050
from fusion.drowsiness_fusion import DrowsinessFusionEngine
from alerts.audio_alert import AudioAlertSystem
from analytics.perclos_classifier import PERCLOSModelClassifier


class WebcamInferenceManager:
    """
    Manages live camera capture, continuous inference, and telemetry emission.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()

        # Core Computer Vision & Analytics Components
        self.face_detector = UnifiedFaceDetector()
        self.eye_predictor = EyeStatePredictor(model_path=config.EYE_MODEL_PATH)
        self.eye_tracker = EyeTracker(default_threshold=config.EAR_THRESHOLD)
        self.perclos_tracker = RollingPERCLOS(
            window_seconds=config.PERCLOS_WINDOW_SECONDS,
            drowsy_threshold=config.PERCLOS_DROWSY_THRESHOLD,
            warning_threshold=config.PERCLOS_WARNING_THRESHOLD,
            sustained_closure_threshold_sec=config.SUSTAINED_CLOSURE_DROWSY_THRESHOLD_SEC,
            max_gap_seconds=config.MAX_TRACKING_GAP_SECONDS
        )
        self.blink_tracker = BlinkTracker()
        self.perclos_model = PERCLOSModelClassifier()
        self.ear_history = deque(maxlen=60)

        # Sensors: Real ESP32 Hardware Bridge & Simulated Fixture
        self.simulated_sensor = SimulatedMPU6050(initial_state="NORMAL")
        self.real_sensor = RealMPU6050(port=config.SERIAL_PORT, baudrate=config.SERIAL_BAUDRATE, auto_start=True)
        self.active_sensor_mode = "REAL"  # Default to REAL hardware per transformation requirement

        # Multi-Signal Fusion Engine
        self.fusion_engine = DrowsinessFusionEngine(
            cnn_threshold=config.CNN_DROWSY_THRESHOLD,
            perclos_drowsy_th=config.PERCLOS_DROWSY_THRESHOLD,
            perclos_warn_th=config.PERCLOS_WARNING_THRESHOLD,
            head_nod_pitch_th=config.HEAD_NOD_PITCH_THRESHOLD,
            head_tilt_roll_th=config.HEAD_TILT_ROLL_THRESHOLD,
            sustained_closure_th=config.SUSTAINED_CLOSURE_DROWSY_THRESHOLD_SEC
        )
        self.audio_alert = AudioAlertSystem(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)

        # CNN Face Model State
        self.model = None
        self.model_status = "NOT_TRAINED"  # "LOADED", "NOT_TRAINED", "ERROR"
        self.model_load_error = None
        self._load_cnn_model()

        # Temporal Smoothing
        self.prediction_history = deque(maxlen=config.PREDICTION_SMOOTHING_WINDOW)

        # Telemetry State
        self.latest_frame_jpeg = None
        self.latest_telemetry = self._build_initial_telemetry()

    def _build_initial_telemetry(self) -> Dict[str, Any]:
        """Generates clean initial telemetry state."""
        return {
            "status": "STOPPED",
            "camera_available": False,
            "model_status": self.model_status,
            "eye_model_status": "LOADED" if self.eye_predictor.is_ready else "MODEL_NOT_READY",
            "fps": 0.0,
            "total_frame_latency_ms": 0.0,
            "face_detected": False,
            "eyes_detected": False,
            "eye_state": {
                "state": "STANDBY",
                "confidence": 0.0,
                "is_closed": False,
                "is_valid": False,
                "model_ready": self.eye_predictor.is_ready
            },
            "ear": {"avg_ear": 0.0, "is_closed": False, "valid": False},
            "perclos": self.perclos_tracker.get_metrics(),
            "blinks": self.blink_tracker.get_metrics(),
            "sensor": self.get_active_sensor().read().to_dict(),
            "sensor_mode": self.active_sensor_mode,
            "cnn": {
                "model_loaded": (self.model_status == "LOADED"),
                "probability": 0.0,
                "predicted_class": "N/A",
                "is_valid": False
            },
            "fusion": {
                "overall_state": "NORMAL",
                "composite_score": 0.0,
                "contributing_factors": ["System Idle / Monitoring Stopped"],
                "confidence_level": "LOW",
                "readiness": self.fusion_engine.evaluate_readiness(
                    camera_operational=False,
                    face_detected=False,
                    eyes_detected=False,
                    eye_model_loaded=self.eye_predictor.is_ready,
                    perclos_valid=False,
                    esp32_connected=self.real_sensor.is_connected(),
                    sensor_valid=False,
                    sensor_verified=False
                )
            },
            "alert": self.audio_alert.get_status(),
            "timestamp": time.time()
        }

    def _load_cnn_model(self):
        if config.MODEL_PATH.exists():
            try:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(config.MODEL_PATH)
                self.model_status = "LOADED"
                self.model_load_error = None
                print(f"[WebcamInference] Trained CNN model loaded: {config.MODEL_PATH}")
            except Exception as e:
                self.model = None
                self.model_status = "ERROR"
                self.model_load_error = str(e)
                print(f"[WebcamInference Error] Failed to load CNN model: {e}")
        else:
            self.model = None
            self.model_status = "NOT_TRAINED"
            print(f"[WebcamInference Notice] Model file not found at {config.MODEL_PATH}. Running in Analytics & Sensor mode.")

    def reload_model(self):
        with self._lock:
            self._load_cnn_model()
            self.eye_predictor._load_model()
            return {
                "face_model_status": self.model_status,
                "eye_model_status": "LOADED" if self.eye_predictor.is_ready else "MODEL_NOT_READY",
                "error": self.model_load_error or self.eye_predictor.load_error
            }

    def set_sensor_mode(self, mode: str):
        with self._lock:
            if mode.upper() in ["SIMULATED", "REAL"]:
                self.active_sensor_mode = mode.upper()
                return True
            return False

    def get_active_sensor(self):
        if self.active_sensor_mode == "REAL":
            return self.real_sensor
        return self.simulated_sensor

    def start(self):
        with self._lock:
            if self.is_running:
                return True, "Monitoring is already running."

            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap or not self.cap.isOpened():
                return False, (
                    "Could not open webcam (index 0). "
                    "Please check camera privacy permissions in Windows Settings -> Privacy & Security -> Camera, "
                    "or ensure no other application is locking the webcam."
                )

            # Warm up camera & configure resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_running = True
            self.perclos_tracker.reset()
            self.blink_tracker.reset()
            self.prediction_history.clear()

            self._thread = threading.Thread(target=self._capture_and_infer_loop, daemon=True, name="WebcamInferenceThread")
            self._thread.start()
            return True, "Webcam monitoring started successfully."

    def stop(self):
        with self._lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

        self.latest_telemetry["status"] = "STOPPED"
        self.latest_telemetry["camera_available"] = False
        return True, "Monitoring stopped and webcam released."

    def _capture_and_infer_loop(self):
        fps_timer = time.time()
        frame_counter = 0
        current_fps = 0.0

        while self.is_running and self.cap and self.cap.isOpened():
            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = round(frame_counter / (time.time() - fps_timer), 1)
                frame_counter = 0
                fps_timer = time.time()

            h_f, w_f = frame.shape[:2]

            # 1. Face & Eye Region Detection
            detection = self.face_detector.detect(frame)
            face_found = detection["face_found"]
            eyes_found = detection.get("eyes_found", False)
            bbox = detection["bbox"]
            landmarks = detection["landmarks"]
            face_crop = detection["face_crop_bgr"]
            l_eye_crop = detection.get("left_eye_crop")
            r_eye_crop = detection.get("right_eye_crop")

            # 2. Eye Aspect Ratio (EAR) Geometry Calculation
            ear_result = {"valid": False, "avg_ear": 0.0, "is_closed": False}
            if face_found and landmarks is not None:
                ear_result = self.eye_tracker.compute_ear(landmarks, w_f, h_f)

            # 3. Real Eye-State CNN Model Inference (Open vs Closed)
            eye_state_result = {
                "state": "MODEL_NOT_READY",
                "confidence": 0.0,
                "closed_probability": 0.0,
                "is_closed": False,
                "is_valid": False,
                "model_ready": self.eye_predictor.is_ready,
                "left_eye": None,
                "right_eye": None
            }

            if face_found and eyes_found and (l_eye_crop is not None or r_eye_crop is not None):
                if self.eye_predictor.is_ready:
                    # Run inference on left and right eye crops
                    preds = []
                    if l_eye_crop is not None:
                        l_pred = self.eye_predictor.predict(l_eye_crop)
                        eye_state_result["left_eye"] = l_pred
                        if l_pred["is_valid"]:
                            preds.append(l_pred["closed_probability"])
                    if r_eye_crop is not None:
                        r_pred = self.eye_predictor.predict(r_eye_crop)
                        eye_state_result["right_eye"] = r_pred
                        if r_pred["is_valid"]:
                            preds.append(r_pred["closed_probability"])

                    if preds:
                        avg_closed_prob = float(np.mean(preds))
                        is_closed_pred = avg_closed_prob >= 0.50
                        eye_state_result["closed_probability"] = round(avg_closed_prob, 3)
                        eye_state_result["is_closed"] = is_closed_pred
                        eye_state_result["state"] = "closed" if is_closed_pred else "open"
                        eye_state_result["confidence"] = round(avg_closed_prob if is_closed_pred else (1.0 - avg_closed_prob), 3)
                        eye_state_result["is_valid"] = True
                else:
                    # Model not trained yet -> report MODEL_NOT_READY honestly
                    eye_state_result["state"] = "MODEL_NOT_READY"
                    eye_state_result["is_valid"] = False

            # 4. Determine authoritative closed-eye observation
            # Robust multi-cue fusion between MediaPipe 3D Landmark EAR and Eye-State CNN:
            now_ts = time.time()
            avg_ear = ear_result.get("avg_ear", 0.0)
            ear_is_closed = ear_result.get("is_closed", False)
            ear_valid = ear_result.get("valid", False)

            if ear_valid and avg_ear >= 0.24:
                # Eye geometry is clearly open (EAR >= 0.24) -> Driver is awake
                authoritative_is_closed = False
                if eye_state_result["is_valid"]:
                    eye_state_result["state"] = "open"
                    eye_state_result["is_closed"] = False
                    eye_state_result["confidence"] = max(0.65, 1.0 - eye_state_result.get("closed_probability", 0.0))
            elif ear_valid and avg_ear < 0.17:
                # Eye geometry is clearly closed (EAR < 0.17)
                authoritative_is_closed = True
                if eye_state_result["is_valid"]:
                    eye_state_result["state"] = "closed"
                    eye_state_result["is_closed"] = True
                    eye_state_result["confidence"] = max(0.65, eye_state_result.get("closed_probability", 0.8))
            elif eye_state_result["is_valid"]:
                # In EAR transition zone (0.17 - 0.24), use trained eye CNN with confident threshold
                authoritative_is_closed = (eye_state_result["closed_probability"] >= 0.60)
                eye_state_result["state"] = "closed" if authoritative_is_closed else "open"
                eye_state_result["is_closed"] = authoritative_is_closed
            else:
                authoritative_is_closed = ear_is_closed

            is_valid_observation = face_found and (ear_valid or eye_state_result["is_valid"])

            # 5. Rolling PERCLOS & Blink Dynamics
            self.perclos_tracker.update(
                is_closed=authoritative_is_closed,
                is_valid=is_valid_observation,
                timestamp=now_ts
            )
            perclos_metrics = self.perclos_tracker.get_metrics()

            blink_event = self.blink_tracker.update(
                is_closed=authoritative_is_closed,
                is_valid=is_valid_observation,
                timestamp=now_ts
            )
            blink_metrics = self.blink_tracker.get_metrics(current_timestamp=now_ts)
            if ear_result.get("valid", False):
                self.ear_history.append(ear_result.get("avg_ear", 0.0))

            # 6. Deep Learning Face-Level CNN Inference (MobileNetV2)
            cnn_result = {
                "model_loaded": (self.model_status == "LOADED"),
                "probability": 0.0,
                "smoothed_probability": 0.0,
                "predicted_class": "N/A",
                "is_valid": False,
                "latency_ms": 0.0
            }

            t_infer_start = time.perf_counter()
            if self.model_status == "LOADED" and self.model is not None and face_found and face_crop is not None:
                try:
                    input_tensor = preprocess_image(face_crop, target_size=config.IMAGE_SIZE)
                    raw_pred = float(self.model(input_tensor, training=False).numpy()[0][0])
                    self.prediction_history.append(raw_pred)
                    smoothed_pred = float(np.mean(self.prediction_history))

                    cnn_result["probability"] = round(raw_pred, 3)
                    cnn_result["smoothed_probability"] = round(smoothed_pred, 3)
                    cnn_result["predicted_class"] = "Drowsy" if smoothed_pred >= config.CNN_DROWSY_THRESHOLD else "Non Drowsy"
                    cnn_result["is_valid"] = True
                except Exception as e:
                    cnn_result["error"] = str(e)
            cnn_result["latency_ms"] = round((time.perf_counter() - t_infer_start) * 1000.0, 1)

            # 7. Read Physical IMU Sensor (Real ESP32 or Simulated)
            sensor = self.get_active_sensor()
            sensor_reading = sensor.read()

            # 8. Multi-Signal Sensor Fusion
            fusion_result = self.fusion_engine.fuse(
                cnn_result=cnn_result,
                perclos_result=perclos_metrics,
                blink_result=blink_metrics,
                sensor_reading=sensor_reading,
                eye_state_result=eye_state_result,
                camera_operational=True,
                face_detected=face_found,
                eyes_detected=eyes_found
            )

            # 9. Audible Alert Triggering with Cooldown
            overall_state = fusion_result["overall_state"]
            if overall_state in ["WARNING", "CRITICAL"]:
                self.audio_alert.trigger(alert_level=overall_state)

            total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

            # 10. Honest Video Annotations for Dashboard Stream
            annotated_frame = frame.copy()
            if face_found and bbox:
                x, y, w, h = bbox
                box_color = (0, 0, 255) if overall_state == "CRITICAL" else ((0, 255, 255) if overall_state == "WARNING" else (0, 255, 0))
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)

                # Eye State Tag
                eye_tag = f"EYES: {eye_state_result['state'].upper()}" if eye_state_result["is_valid"] else f"EAR: {ear_result['avg_ear']:.3f}"
                cv2.putText(annotated_frame, f"{overall_state} | {eye_tag} | PERCLOS: {perclos_metrics['perclos_percent']}%",
                            (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            # Honest HUD Watermark: Reflects real ESP32 connection status
            if sensor_reading.is_connected:
                imu_tag = f"ESP32:{sensor_reading.port} [MPU-6050 VERIFIED]" if sensor_reading.chip_verified else f"ESP32:{sensor_reading.port} [UNVERIFIED]"
                hud_color = (0, 255, 200)
            elif self.active_sensor_mode == "SIMULATED":
                imu_tag = "IMU: SIMULATED FIXTURE"
                hud_color = (200, 200, 0)
            else:
                imu_tag = f"ESP32:{sensor_reading.port} [OFFLINE]"
                hud_color = (150, 150, 150)

            cv2.putText(annotated_frame, f"DMS SENSOR SYNC | {current_fps} FPS | {imu_tag}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)

            # Encode JPEG for video stream
            ret_enc, jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                self.latest_frame_jpeg = jpeg.tobytes()

            # Update Central Telemetry State
            self.latest_telemetry = {
                "status": "RUNNING",
                "camera_available": True,
                "model_status": self.model_status,
                "eye_model_status": "LOADED" if self.eye_predictor.is_ready else "MODEL_NOT_READY",
                "fps": current_fps,
                "total_frame_latency_ms": total_latency_ms,
                "face_detected": face_found,
                "eyes_detected": eyes_found,
                "bbox": bbox,
                "eye_state": eye_state_result,
                "ear": ear_result,
                "perclos": perclos_metrics,
                "blinks": blink_metrics,
                "sensor": sensor_reading.to_dict(),
                "sensor_mode": self.active_sensor_mode,
                "cnn": cnn_result,
                "fusion": fusion_result,
                "alert": self.audio_alert.get_status(),
                "timestamp": now_ts
            }

            time.sleep(0.01)

        # Cleanup on loop exit
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def get_telemetry(self):
        return self.latest_telemetry

    def get_latest_jpeg(self):
        return self.latest_frame_jpeg
