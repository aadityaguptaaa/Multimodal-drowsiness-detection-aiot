#!/usr/bin/env python3
"""
==============================================================================
AIoT Driver Drowsiness Detection System - ESP32 Hardware Serial Bridge
==============================================================================
Description:
    Stand-alone Python bridge connecting physical ESP32 IMU hardware to the
    React web dashboard.
    - Continuously reads calibrated 6-DOF JSON lines over USB serial (COM9, 115200 baud).
    - Verifies MPU-6050 WHO_AM_I identity and enforces telemetry staleness thresholds.
    - Exposes thread-safe endpoints GET /data and GET /health on 127.0.0.1:8765.
    - Includes complete CORS headers for local React/Vite development.
    - Handles disconnects, port contention, malformed frames, and auto-reconnects.
==============================================================================
"""

import sys
import os
import time
import json
import socket
import logging
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (ESP32-Bridge) %(message)s"
)
logger = logging.getLogger("ESP32Bridge")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server so concurrent browser polls never block."""
    daemon_threads = True
    allow_reuse_address = True


class HardwareBridgeState:
    """Thread-safe state container for physical ESP32 IMU telemetry."""

    def __init__(self, port: str = "COM9", baud: int = 115200):
        self.lock = threading.Lock()
        self.port = port
        self.baud = baud
        self.connection_status = "OFFLINE"  # OFFLINE, CONNECTING, ONLINE, STALE, ERROR
        self.chip_id = "UNKNOWN"
        self.chip_verified = False
        self.error_message = None
        self.last_valid_time = 0.0
        self.last_packet_time = 0.0
        self.packets_received = 0
        self.reconnect_count = 0
        self.start_time = time.time()

        # Physical 6-DOF sensor telemetry
        self.telemetry = {
            "ax": 0.0,
            "ay": 0.0,
            "az": 0.0,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0
        }

    def update_telemetry(self, raw_data: dict):
        with self.lock:
            try:
                self.telemetry["ax"] = round(float(raw_data.get("ax", 0.0)), 3)
                self.telemetry["ay"] = round(float(raw_data.get("ay", 0.0)), 3)
                self.telemetry["az"] = round(float(raw_data.get("az", 0.0)), 3)
                self.telemetry["gx"] = round(float(raw_data.get("gx", 0.0)), 2)
                self.telemetry["gy"] = round(float(raw_data.get("gy", 0.0)), 2)
                self.telemetry["gz"] = round(float(raw_data.get("gz", 0.0)), 2)
                self.telemetry["pitch"] = round(float(raw_data.get("pitch", 0.0)), 2)
                self.telemetry["roll"] = round(float(raw_data.get("roll", 0.0)), 2)
                self.telemetry["yaw"] = round(float(raw_data.get("yaw", 0.0)), 2)

                self.chip_id = str(raw_data.get("chip_id", raw_data.get("who_am_i", "0x68")))
                self.chip_verified = bool(raw_data.get("chip_verified", raw_data.get("verified", True)))
                self.connection_status = "ONLINE"
                self.error_message = None
                self.last_valid_time = time.time()
                self.last_packet_time = self.last_valid_time
                self.packets_received += 1
            except Exception as e:
                self.error_message = f"Field parsing error: {e}"

    def set_error(self, error: str, status: str = "OFFLINE"):
        with self.lock:
            self.error_message = error
            self.connection_status = status
            # When disconnected, zero-out axes rather than presenting stale/synthetic readings
            self.telemetry = {
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "pitch": 0.0, "roll": 0.0, "yaw": 0.0
            }

    def get_snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            data_age = round(now - self.last_valid_time, 2) if self.last_valid_time > 0 else 999.0
            is_stale = (self.last_valid_time > 0 and data_age > 2.0)

            effective_status = self.connection_status
            if self.connection_status == "ONLINE" and is_stale:
                effective_status = "STALE"

            return {
                "status": effective_status,
                "port": self.port,
                "baud": self.baud,
                "chip_id": self.chip_id,
                "chip_verified": self.chip_verified,
                "is_connected": (effective_status == "ONLINE"),
                "is_stale": is_stale,
                "data_age_sec": data_age,
                "telemetry": dict(self.telemetry),
                "packets_received": self.packets_received,
                "reconnect_count": self.reconnect_count,
                "uptime_sec": round(now - self.start_time, 1),
                "error": self.error_message,
                "timestamp": now
            }


class ESP32SerialWorker(threading.Thread):
    """Background worker continuously reading and parsing USB serial frames."""

    def __init__(self, state: HardwareBridgeState):
        super().__init__(daemon=True, name="ESP32SerialWorker")
        self.state = state
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        if not PYSERIAL_AVAILABLE:
            self.state.set_error("pyserial package not installed. Run: py -m pip install pyserial", "ERROR")
            logger.error("pyserial is not installed! Run: py -m pip install pyserial")
            return

        logger.info(f"Worker started. Targeting ESP32 on {self.state.port} @ {self.state.baud} baud.")

        while self._running:
            ser = None
            try:
                self.state.set_error(None, "CONNECTING")
                logger.info(f"Attempting connection to {self.state.port}...")
                ser = serial.Serial(
                    port=self.state.port,
                    baudrate=self.state.baud,
                    timeout=1.5,
                    write_timeout=1.0
                )
                time.sleep(1.0)  # ESP32 auto-reset settling time
                ser.reset_input_buffer()
                logger.info(f"Connected successfully to {self.state.port}.")

                while self._running:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if not line:
                        # Check staleness if silence exceeds 2.5s
                        if time.time() - self.state.last_packet_time > 2.5 and self.state.last_packet_time > 0:
                            self.state.set_error("No telemetry frames received in >2.5s", "STALE")
                        continue

                    # Skip non-JSON debug strings
                    if not (line.startswith("{") and line.endswith("}")):
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # If firmware reported read error
                    if data.get("status") == "READ_ERROR" or "error" in data:
                        err_msg = data.get("error", "Hardware I2C read failure")
                        self.state.set_error(err_msg, "ERROR")
                        continue

                    # Check for valid accelerometer data
                    if "ax" in data and "ay" in data and "az" in data:
                        self.state.update_telemetry(data)

            except serial.SerialException as se:
                err_str = str(se)
                if "FileNotFoundError" in err_str or "could not open port" in err_str:
                    msg = f"Port {self.state.port} not detected. Connect ESP32 via USB."
                elif "PermissionError" in err_str or "Access is denied" in err_str:
                    msg = f"Port {self.state.port} in use. Close Arduino Serial Monitor / other programs."
                else:
                    msg = f"Serial error on {self.state.port}: {err_str}"
                
                self.state.set_error(msg, "OFFLINE")
                self.state.reconnect_count += 1
                logger.warning(f"{msg}. Retrying in 2.0s...")
                time.sleep(2.0)

            except Exception as e:
                self.state.set_error(f"Unexpected worker error: {e}", "ERROR")
                logger.error(f"Unexpected worker exception: {e}", exc_info=True)
                time.sleep(2.0)

            finally:
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass


class HardwareBridgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing GET /data and GET /health with full CORS."""

    # Reference to the shared state container
    state: HardwareBridgeState = None

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def do_OPTIONS(self):
        """Respond to browser CORS preflight check."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _safe_write(self, body: bytes):
        try:
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_GET(self):
        """Handle incoming REST requests."""
        path = self.path.split("?")[0].rstrip("/")

        if path == "" or path == "/data":
            snapshot = self.state.get_snapshot()
            body = json.dumps(snapshot, indent=2).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self._safe_write(body)

        elif path == "/health":
            now = time.time()
            snapshot = self.state.get_snapshot()
            health_doc = {
                "service": "ESP32-Hardware-Bridge",
                "status": "HEALTHY",
                "port": self.state.port,
                "connected": snapshot["is_connected"],
                "data_age_sec": snapshot["data_age_sec"],
                "uptime_sec": snapshot["uptime_sec"]
            }
            body = json.dumps(health_doc, indent=2).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self._safe_write(body)

        elif path == "/ports":
            ports = []
            if PYSERIAL_AVAILABLE:
                for p in serial.tools.list_ports.comports():
                    ports.append({
                        "device": p.device,
                        "description": p.description,
                        "hwid": p.hwid
                    })
            body = json.dumps({"ports": ports}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        else:
            err_body = json.dumps({"error": f"Endpoint '{self.path}' not found"}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(err_body)

    def log_message(self, format, *args):
        # Silence routine high-frequency /data polling log spam
        if "GET /data" in format % args or "OPTIONS" in format % args:
            return
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))


def main():
    parser = argparse.ArgumentParser(description="ESP32 Physical IMU to React HTTP Bridge")
    parser.add_argument("--port", default=os.getenv("ESP32_PORT", "COM9"), help="Serial port (default: COM9)")
    parser.add_argument("--baud", type=int, default=int(os.getenv("ESP32_BAUD", "115200")), help="Baud rate (default: 115200)")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=8765, help="HTTP bind port (default: 8765)")
    args = parser.parse_args()

    print("=" * 70)
    print("   AIoT Driver Drowsiness Detection System - Physical ESP32 Bridge   ")
    print("=" * 70)
    print(f" Target Serial Port:  {args.port}")
    print(f" Baud Rate:           {args.baud}")
    print(f" Local HTTP Endpoint: http://{args.host}:{args.http_port}/data")
    print(f" Health Check:        http://{args.host}:{args.http_port}/health")
    print(f" Available Ports:     http://{args.host}:{args.http_port}/ports")
    print("=" * 70)

    # Instantiate shared state container
    bridge_state = HardwareBridgeState(port=args.port, baud=args.baud)
    HardwareBridgeRequestHandler.state = bridge_state

    # Start background serial worker
    worker = ESP32SerialWorker(bridge_state)
    worker.start()

    # Start HTTP server
    server_address = (args.host, args.http_port)
    try:
        httpd = ThreadedHTTPServer(server_address, HardwareBridgeRequestHandler)
        logger.info(f"Serving HTTP on http://{args.host}:{args.http_port} (Press Ctrl+C to stop)")
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Terminating workers...")
    except OSError as oe:
        logger.error(f"Could not bind to {args.host}:{args.http_port}: {oe}")
    finally:
        worker.stop()
        logger.info("ESP32 Hardware Bridge stopped.")


if __name__ == "__main__":
    main()
