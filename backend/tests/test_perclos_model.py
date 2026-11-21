"""
Automated Unit Tests for PERCLOS Machine Learning Model & Classifier
"""

import sys
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics.perclos_classifier import PERCLOSModelClassifier

def test_perclos_model_loading_and_prediction():
    classifier = PERCLOSModelClassifier()
    assert classifier.is_loaded is True

    # Alert inputs: low PERCLOS (0.05), open EAR (0.32), normal blinks (15 BPM)
    alert_metrics = {
        "valid_observation": True,
        "perclos_ratio": 0.04,
        "closed_duration_sec": 0.4
    }
    ear_history = [0.32, 0.31, 0.33, 0.30, 0.32]
    blink_metrics = {"blinks_per_minute": 15.0, "average_duration_ms": 210.0}

    pred_alert = classifier.predict(alert_metrics, ear_history, blink_metrics)
    assert pred_alert["is_valid"] is True
    assert pred_alert["is_drowsy"] is False
    assert pred_alert["predicted_class"] == "Alert"

    # Drowsy inputs: high PERCLOS (0.45), drooping EAR (0.16), sluggish blinks
    drowsy_metrics = {
        "valid_observation": True,
        "perclos_ratio": 0.45,
        "closed_duration_sec": 4.5
    }
    ear_history_drowsy = [0.16, 0.15, 0.14, 0.17, 0.15]
    blink_metrics_drowsy = {"blinks_per_minute": 4.0, "average_duration_ms": 520.0}

    pred_drowsy = classifier.predict(drowsy_metrics, ear_history_drowsy, blink_metrics_drowsy)
    assert pred_drowsy["is_valid"] is True
    assert pred_drowsy["is_drowsy"] is True
    assert pred_drowsy["predicted_class"] == "Drowsy"
