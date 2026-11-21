"""
Unit Tests for Eye-State Classifier Model & Preprocessing Pipeline
"""

import sys
import numpy as np
import pytest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from training.train_eye_model import build_lightweight_eye_model
from inference.predict_eye import EyeStatePredictor


def test_eye_model_architecture():
    """Validates lightweight eye model construction, input shape, and parameter budget."""
    model = build_lightweight_eye_model(input_shape=(64, 64, 3))
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, 1)

    # Validate model is lightweight (< 200,000 parameters for edge deployment)
    total_params = model.count_params()
    assert total_params < 200000, f"Eye model has {total_params} parameters, exceeding 200k edge budget"


def test_eye_predictor_missing_model_behavior():
    """Validates that EyeStatePredictor returns MODEL_NOT_READY when model is absent, never faking predictions."""
    dummy_path = BACKEND_DIR / "models" / "non_existent_eye_model.keras"
    predictor = EyeStatePredictor(model_path=dummy_path)
    assert predictor.is_ready is False

    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = predictor.predict(dummy_image)

    assert result["state"] == "MODEL_NOT_READY"
    assert result["is_valid"] is False
    assert result["confidence"] == 0.0
    assert result["error"] is not None


def test_eye_predictor_preprocessing_shape_and_range():
    """Validates that preprocessing produces (1, 64, 64, 3) float32 in [-1, 1]."""
    dummy_path = BACKEND_DIR / "models" / "non_existent_eye_model.keras"
    predictor = EyeStatePredictor(model_path=dummy_path)

    # Test random uint8 image [0, 255]
    img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    processed = predictor.preprocess(img)

    assert processed.shape == (1, 64, 64, 3)
    assert processed.dtype == np.float32
    assert np.min(processed) >= -1.05
    assert np.max(processed) <= 1.05
