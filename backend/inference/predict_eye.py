"""
Reproducible Inference Script for Eye-State Model
Evaluates eye openness on standalone images or image arrays.

Usage from CLI:
    python predict_eye.py path/to/eye_sample.jpg

Usage from Python:
    from inference.predict_eye import EyeStatePredictor
    predictor = EyeStatePredictor()
    result = predictor.predict(eye_image_bgr)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import cv2

# Suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    keras = None


class EyeStatePredictor:
    """
    Inference wrapper for the trained eye-state classifier.
    Preprocesses input eye image to 64x64 and [-1, 1] normalization,
    matching training conditions.
    """
    def __init__(self, model_path: Union[str, Path] = config.EYE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.is_ready = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        if not HAS_TF:
            self.load_error = "TensorFlow not installed"
            self.is_ready = False
            return

        if self.model_path.exists():
            try:
                self.model = keras.models.load_model(self.model_path)
                self.is_ready = True
                self.load_error = None
            except Exception as e:
                self.model = None
                self.is_ready = False
                self.load_error = str(e)
        else:
            self.model = None
            self.is_ready = False
            self.load_error = f"Model artifact not found at: {self.model_path}"

    def preprocess(self, eye_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Loads and prepares an eye crop for inference.
        Returns tensor of shape (1, 64, 64, 3) normalized to [-1, 1].
        """
        if isinstance(eye_input, (str, Path)):
            img = cv2.imread(str(eye_input))
            if img is None:
                raise ValueError(f"Unable to read image at: {eye_input}")
        elif isinstance(eye_input, np.ndarray):
            img = eye_input
        else:
            raise TypeError(f"Unsupported image type: {type(eye_input)}")

        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 64x64
        resized = cv2.resize(img, config.EYE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        # Normalize [-1, 1]
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, eye_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Runs model inference.
        Returns:
            dict containing:
            - state: "open" or "closed" (or "MODEL_NOT_READY")
            - confidence: float [0, 1]
            - closed_probability: float [0, 1]
            - is_closed: bool
            - is_valid: bool
            - error: Optional[str]
        """
        if not self.is_ready or self.model is None:
            return {
                "state": "MODEL_NOT_READY",
                "confidence": 0.0,
                "closed_probability": 0.0,
                "is_closed": False,
                "is_valid": False,
                "error": self.load_error or "Eye state model not trained/loaded"
            }

        try:
            tensor = self.preprocess(eye_input)
            raw_prob = float(self.model(tensor, training=False).numpy()[0][0])
            is_closed = raw_prob >= 0.50
            confidence = raw_prob if is_closed else (1.0 - raw_prob)

            return {
                "state": "closed" if is_closed else "open",
                "confidence": round(confidence, 3),
                "closed_probability": round(raw_prob, 3),
                "is_closed": is_closed,
                "is_valid": True,
                "error": None
            }
        except Exception as e:
            return {
                "state": "ERROR",
                "confidence": 0.0,
                "closed_probability": 0.0,
                "is_closed": False,
                "is_valid": False,
                "error": str(e)
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIoT Eye State Predictor")
    parser.add_argument("image_path", type=str, help="Path to eye image file")
    parser.add_argument("--model-path", type=str, default=str(config.EYE_MODEL_PATH), help="Custom model path")
    args = parser.parse_args()

    predictor = EyeStatePredictor(model_path=args.model_path)
    res = predictor.predict(args.image_path)
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Predicted Eye State: {res['state'].upper()}")
    print(f"Confidence:          {res['confidence'] * 100:.1f}%")
    print(f"Closed Probability:  {res['closed_probability']:.3f}")
    if res["error"]:
        print(f"Error Notice:        {res['error']}")
    print("=" * 60)
