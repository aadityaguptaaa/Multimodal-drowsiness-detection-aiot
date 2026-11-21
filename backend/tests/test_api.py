"""
Automated Integration Tests for Flask REST Endpoints
"""

import sys
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app, webcam_manager

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_health(client):
    """Tests GET /api/health returns 200 with HEALTHY status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "HEALTHY"
    assert "python_version" in data

def test_api_dataset_inspect(client):
    """Tests GET /api/dataset/inspect returns valid dataset metadata."""
    res = client.get("/api/dataset/inspect")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "VALID"
    assert data["total_images"] > 0

def test_api_model_status(client):
    """Tests GET /api/model/status returns trained model status."""
    res = client.get("/api/model/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["model_status"] in ["LOADED", "NOT_TRAINED"]
    assert data["model_file_exists"] is True

def test_api_monitoring_telemetry(client):
    """Tests GET /api/monitoring/telemetry returns full telemetry packet."""
    res = client.get("/api/monitoring/telemetry")
    assert res.status_code == 200
    data = res.get_json()
    assert "ear" in data
    assert "perclos" in data
    assert "fusion" in data

def test_api_sensor_simulation_controls(client):
    """Tests POST /api/sensors/simulate sets simulator states correctly."""
    res = client.post("/api/sensors/simulate", json={"state": "HEAD_NOD"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["state"] == "HEAD_NOD"

    # Invalid state rejection
    bad_res = client.post("/api/sensors/simulate", json={"state": "NON_EXISTENT_STATE"})
    assert bad_res.status_code == 400
