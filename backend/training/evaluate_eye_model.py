"""
Held-Out Test Set Evaluation Pipeline for Eye-State Classification Model
Computes rigorous test set metrics:
- Accuracy, Precision, Recall, F1-Score (macro & per-class: open, closed)
- Confusion Matrix (True Open, False Closed, False Open, True Closed)
- Latency & Throughput Benchmark (ms/eye crop, FPS)
- Outputs report JSON to backend/models/eye_evaluation_report.json
- Renders high-resolution confusion matrix heatmap to backend/models/eye_confusion_matrix.png
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from training.train_eye_model import load_and_preprocess_sample

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


def plot_eye_confusion_matrix(cm, class_names, save_path):
    """Renders and saves confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Eye State Model: Held-Out Test Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    plt.ylabel("True Class (0=Open, 1=Closed)", fontsize=10, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def measure_eye_latency(model, sample_shape=(*config.EYE_IMAGE_SIZE, 3), iterations=100):
    """Measures warm-up and average inference latency per eye."""
    dummy = np.random.uniform(-1.0, 1.0, size=(1, *sample_shape)).astype(np.float32)
    # Warm-up
    for _ in range(10):
        _ = model(dummy, training=False)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    fps = round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0
    return {
        "average_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "approximate_fps": fps,
        "iterations": iterations
    }


def evaluate_eye_model(model_path=None, test_split_path=None):
    """Runs evaluation on held-out test split."""
    if model_path is None:
        model_path = config.EYE_MODEL_PATH
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"[ERROR] Eye model not found at: {model_path}. Train it first.")
        return {"status": "ERROR", "message": f"Model not found at {model_path}"}

    if test_split_path is None:
        test_split_path = config.MODELS_DIR / "eye_test_split.json"
    test_split_path = Path(test_split_path)

    if not test_split_path.exists():
        print(f"[ERROR] Test split metadata not found at: {test_split_path}. Run train_eye_model.py.")
        return {"status": "ERROR", "message": f"Test split not found at {test_split_path}"}

    with open(test_split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_paths = split_data["test_paths"]
    test_labels = split_data["test_labels"]

    print("=" * 70)
    print("  AIoT EYE STATE CLASSIFIER HELD-OUT TEST EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test Samples: {len(test_paths)} (Open: {test_labels.count(0)}, Closed: {test_labels.count(1)})")

    model = keras.models.load_model(model_path)

    # Process and batch test samples
    y_true = np.array(test_labels, dtype=int)
    y_probs = []

    print("Running inference across held-out test set...")
    for p in test_paths:
        img_bytes = tf.io.read_file(p)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, config.EYE_IMAGE_SIZE, method="area")
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        batched = tf.expand_dims(img, axis=0)
        prob = float(model(batched, training=False).numpy()[0][0])
        y_probs.append(prob)

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.50).astype(int)

    # Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Benchmarks
    latency_info = measure_eye_latency(model)

    report_data = {
        "status": "COMPLETED",
        "model_file": model_path.name,
        "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "test_samples_count": len(test_paths),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "per_class_metrics": {
            "open": {
                "precision": round(float(per_class_p[0]), 4),
                "recall": round(float(per_class_r[0]), 4),
                "f1_score": round(float(per_class_f1[0]), 4),
                "test_count": int(per_class_supp[0])
            },
            "closed": {
                "precision": round(float(per_class_p[1]), 4),
                "recall": round(float(per_class_r[1]), 4),
                "f1_score": round(float(per_class_f1[1]), 4),
                "test_count": int(per_class_supp[1])
            }
        },
        "confusion_matrix": {
            "true_open_correct": int(cm[0, 0]),
            "false_closed_error": int(cm[0, 1]),
            "false_open_missed": int(cm[1, 0]),
            "true_closed_caught": int(cm[1, 1])
        },
        "latency_benchmark": latency_info,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save outputs
    with open(config.EYE_EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    plot_eye_confusion_matrix(cm, ["Open", "Closed"], config.EYE_CONFUSION_MATRIX_PATH)

    print("\nEvaluation Results:")
    print(f" - Test Accuracy:  {accuracy * 100:.2f}%")
    print(f" - Macro F1-Score: {f1:.4f}")
    print(f" - Open Eye F1:    {per_class_f1[0]:.4f}")
    print(f" - Closed Eye F1:  {per_class_f1[1]:.4f}")
    print(f" - Avg Latency:    {latency_info['average_latency_ms']} ms/eye ({latency_info['approximate_fps']} FPS)")
    print(f"Report saved to: {config.EYE_EVALUATION_REPORT_PATH}")
    return report_data


if __name__ == "__main__":
    evaluate_eye_model()
