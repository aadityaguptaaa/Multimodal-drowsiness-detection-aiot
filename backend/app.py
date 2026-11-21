"""
Flask & Socket.IO Real-Time Backend Service for AIoT Driver Drowsiness Detection
Provides REST endpoints and WebSocket channels for the frontend monitoring dashboard.
"""

import os
import sys
import json
import time
from pathlib import Path
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from inference.webcam import WebcamInferenceManager

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "aiot_drowsiness_2026_dev_key")
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize central webcam & inference orchestrator
webcam_manager = WebcamInferenceManager()

# Event log history (in-memory ring buffer)
event_history = []
MAX_EVENT_HISTORY = 100

def record_event(level, message, score=0.0):
    event = {
        "id": len(event_history) + 1,
        "timestamp": time.strftime("%H:%M:%S"),
        "epoch": time.time(),
        "level": level,  # "INFO", "WARNING", "CRITICAL"
        "message": message,
        "score": score
    }
    event_history.insert(0, event)
    if len(event_history) > MAX_EVENT_HISTORY:
        event_history.pop()
    socketio.emit("new_event", event)

# Background telemetry broadcasting thread
telemetry_thread = None
telemetry_thread_lock = threading_lock = False

def background_telemetry_emitter():
    last_state = "NORMAL"
    while True:
        try:
            if webcam_manager.is_running:
                data = webcam_manager.get_telemetry()
                socketio.emit("telemetry_update", data)

                # Check state changes for event logging
                current_state = data.get("fusion", {}).get("overall_state", "NORMAL")
                if current_state != last_state:
                    if current_state == "CRITICAL":
                        reasons = ", ".join(data.get("fusion", {}).get("contributing_factors", ["Drowsiness detected"]))
                        record_event("CRITICAL", f"Critical Alert Triggered: {reasons}", data.get("fusion", {}).get("composite_score", 0))
                    elif current_state == "WARNING":
                        reasons = ", ".join(data.get("fusion", {}).get("contributing_factors", ["Fatigue signs"]))
                        record_event("WARNING", f"Fatigue Warning: {reasons}", data.get("fusion", {}).get("composite_score", 0))
                    elif current_state == "NORMAL" and last_state in ["WARNING", "CRITICAL"]:
                        record_event("INFO", "Driver state returned to Normal", 0)
                    last_state = current_state

            time.sleep(0.15)  # ~6.6 Hz updates to dashboard
        except Exception as e:
            time.sleep(0.5)

@app.before_request
def start_background_emitter():
    global telemetry_thread, telemetry_thread_lock
    if not telemetry_thread_lock:
        import threading
        telemetry_thread_lock = True
        telemetry_thread = threading.Thread(target=background_telemetry_emitter, daemon=True)
        telemetry_thread.start()

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "HEALTHY",
        "service": "AIoT Driver Drowsiness Detection Backend",
        "python_version": sys.version,
        "monitoring_running": webcam_manager.is_running,
        "model_status": webcam_manager.model_status,
        "timestamp": time.time()
    })

@app.route("/api/dataset/inspect", methods=["GET"])
def get_dataset_report():
    if config.DATASET_REPORT_PATH.exists():
        with open(config.DATASET_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({
        "status": "NOT_INSPECTED",
        "message": "Dataset has not been inspected yet. Run inspect_dataset.py."
    }), 404

@app.route("/api/model/status", methods=["GET"])
def get_model_status():
    metadata = {}
    if config.METADATA_PATH.exists():
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return jsonify({
        "model_status": webcam_manager.model_status,
        "model_file_exists": config.MODEL_PATH.exists(),
        "model_path": str(config.MODEL_PATH),
        "model_size_mb": round(os.path.getsize(config.MODEL_PATH) / (1024 * 1024), 2) if config.MODEL_PATH.exists() else 0,
        "metadata": metadata,
        "load_error": webcam_manager.model_load_error
    })

@app.route("/api/model/evaluate", methods=["GET"])
def get_evaluation_report():
    if config.EVALUATION_REPORT_PATH.exists():
        with open(config.EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({
        "status": "NOT_EVALUATED",
        "message": "Model has not been evaluated on test split yet. Run evaluate_model.py."
    }), 404

@app.route("/api/model/perclos", methods=["GET"])
def get_perclos_model_report():
    perclos_json = config.MODELS_DIR / "perclos_evaluation_report.json"
    if perclos_json.exists():
        with open(perclos_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({
        "status": "NOT_TRAINED",
        "message": "PERCLOS model not trained yet. Run train_perclos_model.py."
    }), 404

@app.route("/api/model/eye/status", methods=["GET"])
def get_eye_model_status():
    metadata = {}
    if config.EYE_METADATA_PATH.exists():
        with open(config.EYE_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return jsonify({
        "eye_model_status": "LOADED" if webcam_manager.eye_predictor.is_ready else "MODEL_NOT_READY",
        "model_file_exists": config.EYE_MODEL_PATH.exists(),
        "model_path": str(config.EYE_MODEL_PATH),
        "model_size_mb": round(os.path.getsize(config.EYE_MODEL_PATH) / (1024 * 1024), 2) if config.EYE_MODEL_PATH.exists() else 0,
        "metadata": metadata,
        "load_error": webcam_manager.eye_predictor.load_error
    })

@app.route("/api/model/eye/evaluate", methods=["GET"])
def get_eye_model_evaluation():
    if config.EYE_EVALUATION_REPORT_PATH.exists():
        with open(config.EYE_EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({
        "status": "NOT_EVALUATED",
        "message": "Eye model has not been evaluated on test split yet. Run evaluate_eye_model.py."
    }), 404

@app.route("/api/sensors/status", methods=["GET"])
def get_sensor_hardware_status():
    diag = webcam_manager.real_sensor.get_diagnostics()
    diag["active_sensor_mode"] = webcam_manager.active_sensor_mode
    return jsonify(diag)

@app.route("/api/sensors/ports", methods=["GET"])
def get_available_serial_ports():
    from sensors.esp32_bridge import ESP32SerialBridge
    ports = ESP32SerialBridge.list_available_ports()
    return jsonify({"ports": ports, "default_port": config.SERIAL_PORT})

@app.route("/api/sensors/configure", methods=["POST"])
def configure_sensor_hardware():
    data = request.get_json(silent=True) or {}
    port = data.get("port", config.SERIAL_PORT)
    baud = int(data.get("baudrate", config.SERIAL_BAUDRATE))
    webcam_manager.real_sensor.set_port(port, baud)
    record_event("INFO", f"ESP32 Serial port reconfigured to {port} @ {baud} baud")
    return jsonify({
        "status": "SUCCESS",
        "configured_port": port,
        "configured_baud": baud,
        "connected": webcam_manager.real_sensor.is_connected()
    })

@app.route("/api/monitoring/start", methods=["POST"])
def start_monitoring():
    success, message = webcam_manager.start()
    if success:
        record_event("INFO", "Camera monitoring started by user")
        return jsonify({"status": "SUCCESS", "message": message})
    return jsonify({"status": "ERROR", "message": message}), 400

@app.route("/api/monitoring/stop", methods=["POST"])
def stop_monitoring():
    success, message = webcam_manager.stop()
    record_event("INFO", "Camera monitoring stopped by user")
    return jsonify({"status": "SUCCESS", "message": message})

@app.route("/api/monitoring/telemetry", methods=["GET"])
def get_telemetry():
    return jsonify(webcam_manager.get_telemetry())

@app.route("/api/camera/stream", methods=["GET"])
def camera_stream():
    """MJPEG continuous video stream endpoint with proper headers."""
    def generate_stream():
        # Wait up to 3 seconds for camera warm-up and first frame
        wait_start = time.time()
        while webcam_manager.is_running and webcam_manager.get_latest_jpeg() is None and (time.time() - wait_start < 3.0):
            time.sleep(0.05)

        while webcam_manager.is_running:
            jpeg_bytes = webcam_manager.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
            time.sleep(0.04)

    response = Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/api/camera/snapshot", methods=["GET"])
def camera_snapshot():
    jpeg_bytes = webcam_manager.get_latest_jpeg()
    if jpeg_bytes is not None:
        return Response(jpeg_bytes, mimetype="image/jpeg")
    return jsonify({"error": "No frame available"}), 404

@app.route("/api/sensors/mode", methods=["POST"])
def set_sensor_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "SIMULATED").upper()
    success = webcam_manager.set_sensor_mode(mode)
    if success:
        record_event("INFO", f"Sensor mode switched to {mode}")
        return jsonify({"status": "SUCCESS", "sensor_mode": mode})
    return jsonify({"status": "ERROR", "message": "Invalid mode. Choose SIMULATED or REAL"}), 400

@app.route("/api/sensors/simulate", methods=["POST"])
def set_sensor_simulation_state():
    data = request.get_json(silent=True) or {}
    state = data.get("state", "NORMAL").upper()
    try:
        if state == "CUSTOM":
            pitch = data.get("pitch", 0.0)
            roll = data.get("roll", 0.0)
            webcam_manager.simulated_sensor.set_custom_angles(pitch, roll)
        else:
            webcam_manager.simulated_sensor.set_state(state)
        record_event("INFO", f"Simulated MPU6050 state changed to {state}")
        return jsonify({"status": "SUCCESS", "state": state})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route("/api/alert/toggle", methods=["POST"])
def toggle_alert():
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", True)
    webcam_manager.audio_alert.set_enabled(enabled)
    record_event("INFO", f"Audible alert {'enabled' if enabled else 'muted'}")
    return jsonify({"status": "SUCCESS", "alert_enabled": enabled})

@app.route("/api/alert/test", methods=["POST"])
def test_alert():
    triggered = webcam_manager.audio_alert.test_alert()
    record_event("INFO", "Audible alert test beep triggered")
    return jsonify({"status": "SUCCESS", "triggered": triggered})

@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"events": event_history})

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    global event_history
    event_history = []
    return jsonify({"status": "SUCCESS", "message": "Event history cleared"})

if __name__ == "__main__":
    print(f"Starting AIoT Drowsiness Detection Backend on {config.API_HOST}:{config.API_PORT}...")
    socketio.run(app, host=config.API_HOST, port=config.API_PORT, debug=False, allow_unsafe_werkzeug=True)
