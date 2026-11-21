"""
PERCLOS Machine Learning Inference Module
Loads the trained perclos_model.joblib and makes continuous predictions on live rolling features.
"""

import sys
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
import joblib

class PERCLOSModelClassifier:
    """
    Evaluates rolling visual features using the trained SVM/RandomForest PERCLOS model.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODELS_DIR / "perclos_model.joblib"
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                print(f"[PERCLOSModel] Trained PERCLOS model loaded successfully from {self.model_path}")
            except Exception as e:
                self.model = None
                self.is_loaded = False
                print(f"[PERCLOSModel Error] Failed to load {self.model_path}: {e}")
        else:
            self.model = None
            self.is_loaded = False

    def predict(self, perclos_metrics, ear_history, blink_metrics):
        """
        Extracts features from rolling buffers and returns model prediction.
        
        Args:
            perclos_metrics: dict from RollingPERCLOS.get_metrics()
            ear_history: list or deque of recent EAR floats
            blink_metrics: dict from BlinkTracker.get_metrics()
            
        Returns:
            dict containing:
            - is_valid: bool
            - probability: float [0, 1]
            - predicted_class: 'Drowsy' or 'Alert'
            - is_drowsy: bool
        """
        if not self.is_loaded or self.model is None or not perclos_metrics.get("valid_observation", False):
            return {
                "is_valid": False,
                "probability": 0.0,
                "predicted_class": "N/A",
                "is_drowsy": False,
                "model_loaded": self.is_loaded
            }

        perclos_ratio = float(perclos_metrics.get("perclos_ratio", 0.0))
        closed_dur = float(perclos_metrics.get("closed_duration_sec", perclos_ratio * 10.0))

        if ear_history and len(ear_history) > 0:
            ear_arr = np.array(ear_history, dtype=np.float32)
            mean_ear = float(np.mean(ear_arr))
            min_ear = float(np.min(ear_arr))
            ear_var = float(np.var(ear_arr))
        else:
            mean_ear = 0.28
            min_ear = 0.20
            ear_var = 0.002

        bpm = float(blink_metrics.get("blinks_per_minute", 16.0)) if blink_metrics else 16.0
        avg_dur = float(blink_metrics.get("average_duration_ms", 200.0)) if blink_metrics else 200.0
        longest_closure = closed_dur if closed_dur > 0 else 0.2

        feature_vector = np.array([[
            perclos_ratio,
            mean_ear,
            min_ear,
            closed_dur,
            bpm,
            avg_dur,
            longest_closure,
            ear_var
        ]], dtype=np.float32)

        try:
            prob = float(self.model.predict_proba(feature_vector)[0][1])
            is_drowsy = prob >= 0.50
            return {
                "is_valid": True,
                "probability": round(prob, 3),
                "predicted_class": "Drowsy" if is_drowsy else "Alert",
                "is_drowsy": is_drowsy,
                "model_loaded": True
            }
        except Exception as e:
            return {
                "is_valid": False,
                "probability": 0.0,
                "predicted_class": "Error",
                "is_drowsy": False,
                "model_loaded": True,
                "error": str(e)
            }
