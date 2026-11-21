"""
Held-Out Test Set Evaluation Pipeline for Driver Drowsiness Model
Computes rigorous test metrics, confusion matrix, latency benchmarks, and safety failure mode analysis.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from inference.preprocessing import preprocess_image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Renders and saves a high-resolution confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Held-Out Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")

def measure_inference_latency(model, sample_shape=config.INPUT_SHAPE, num_iterations=100):
    """
    Measures warm-up and average per-frame inference latency (ms/frame) and FPS on current machine.
    """
    dummy_input = np.random.uniform(-1.0, 1.0, size=(1, *sample_shape)).astype(np.float32)
    
    # Warm-up iterations
    for _ in range(15):
        _ = model(dummy_input, training=False)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

    return {
        "average_latency_ms": round(avg_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "approximate_fps": fps,
        "benchmark_iterations": num_iterations
    }

def evaluate_model(model_path=None, test_split_path=None):
    """
    Evaluates the model on the held-out test split.
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    model_path = Path(model_path)

    if not model_path.exists():
        msg = f"[ERROR] Model file not found at: {model_path}. Please train the model first."
        print(msg)
        return {"status": "ERROR", "message": msg}

    if test_split_path is None:
        test_split_path = config.MODELS_DIR / "test_split.json"
    test_split_path = Path(test_split_path)

    if not test_split_path.exists():
        msg = f"[ERROR] Test split metadata not found at: {test_split_path}."
        print(msg)
        return {"status": "ERROR", "message": msg}

    print("=" * 70)
    print("  AIoT HELD-OUT TEST SET EVALUATION")
    print("=" * 70)
    print(f"Loading Model: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

    with open(test_split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_paths = split_data["test_paths"]
    test_labels = np.array(split_data["test_labels"])
    class_names = ["Non Drowsy", "Drowsy"]

    print(f"Evaluating on {len(test_paths)} held-out test images...")

    # Batch prediction to prevent out-of-memory
    y_probs = []
    batch_size = 32
    for i in range(0, len(test_paths), batch_size):
        batch_paths = test_paths[i:i + batch_size]
        batch_images = []
        for p in batch_paths:
            img_arr = preprocess_image(p, target_size=config.IMAGE_SIZE)
            batch_images.append(img_arr[0])
        batch_tensor = np.array(batch_images, dtype=np.float32)
        preds = model(batch_tensor, training=False).numpy().flatten()
        y_probs.extend(preds)

    y_probs = np.array(y_probs)
    y_preds = (y_probs >= 0.50).astype(int)

    # 1. Compute Metrics
    acc = float(accuracy_score(test_labels, y_preds))
    precision, recall, f1, support = precision_recall_fscore_support(test_labels, y_preds, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(test_labels, y_preds, average="macro")

    cm = confusion_matrix(test_labels, y_preds)
    tn, fp, fn, tp = cm.ravel()

    # 2. Measure Hardware Latency
    latency_info = measure_inference_latency(model)
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)

    # 3. Safety Analysis of Failure Points
    safety_analysis = {
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_negative_safety_impact": (
            "CRITICAL: A False Negative occurs when a drowsy driver is misclassified as alert. "
            "In road-safety systems, this is the most dangerous failure mode as it fails to trigger alerts. "
            "Our secondary visual signals (Rolling PERCLOS and Blink Tracker) and MPU6050 head tilt fusion "
            "specifically guard against this failure point."
        ),
        "false_positive_convenience_impact": (
            "MODERATE: A False Positive flags an alert driver as drowsy (nuisance alarm). "
            "Excessive false positives cause alarm fatigue, which our temporal smoothing and 10s PERCLOS window mitigate."
        ),
        "domain_shift_notice": (
            "Trained on RGB images from DDD. Transitioning to future IR camera or night-time conditions "
            "will require calibration or domain adaptation as documented in the research presentation."
        )
    }

    # 4. Generate Confusion Matrix Plot
    plot_confusion_matrix(cm, class_names, config.CONFUSION_MATRIX_PATH)

    # 5. Build Report
    report = {
        "status": "COMPLETED",
        "model_file": str(model_path.name),
        "model_size_mb": model_size_mb,
        "test_samples_count": len(test_labels),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "per_class_metrics": {
            "Non Drowsy": {
                "precision": round(float(precision[0]), 4),
                "recall": round(float(recall[0]), 4),
                "f1_score": round(float(f1[0]), 4),
                "test_count": int(support[0])
            },
            "Drowsy": {
                "precision": round(float(precision[1]), 4),
                "recall": round(float(recall[1]), 4),
                "f1_score": round(float(f1[1]), 4),
                "test_count": int(support[1])
            }
        },
        "confusion_matrix": {
            "true_negative_alert_correct": int(tn),
            "false_positive_alert_wrong": int(fp),
            "false_negative_drowsy_missed": int(fn),
            "true_positive_drowsy_caught": int(tp)
        },
        "latency_benchmark": latency_info,
        "safety_failure_analysis": safety_analysis
    }

    with open(config.EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1 * 100:.2f}%")
    print(f"Drowsy Recall:    {recall[1] * 100:.2f}% (Drowsy events correctly identified)")
    print(f"Inference Latency: {latency_info['average_latency_ms']} ms/frame (~{latency_info['approximate_fps']} FPS)")
    print(f"Model File Size:  {model_size_mb} MB")
    print(f"Evaluation report written to: {config.EVALUATION_REPORT_PATH}")
    print("=" * 70)

    return report

if __name__ == "__main__":
    evaluate_model()
