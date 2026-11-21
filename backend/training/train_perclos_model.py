"""
PERCLOS Machine Learning Training Pipeline
Implements the PERCLOS-based classification approach from literature survey (Paper #8: Kumar et al., 2023).
Uses visual behavior features:
- Rolling PERCLOS (proportion of eye closure over window)
- Mean and Minimum EAR
- Eye closure duration
- Blink rate (BPM)
- Average blink duration
- Longest contiguous closure (microsleep indicator)
- EAR variance

Trains an SVM (with RBF kernel) and Random Forest classifier.
Evaluates on a held-out test set and saves the model to backend/models/perclos_model.joblib.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_NAMES = [
    "perclos_ratio",             # [0, 1]
    "mean_ear",                  # ~0.10 to 0.40
    "min_ear",                   # ~0.05 to 0.35
    "closed_duration_sec",       # [0, 10.0]
    "blinks_per_minute",         # [0, 40]
    "avg_blink_duration_ms",     # [100, 800]
    "longest_closure_sec",       # [0, 6.0]
    "ear_variance"               # [0.0001, 0.05]
]

def generate_perclos_dataset(n_samples=2500, random_seed=config.RANDOM_SEED):
    """
    Constructs a calibrated visual-behavior dataset based on empirical EAR parameters
    and literature distributions from RLDD / DDD.
    
    Generates:
    - Alert Class (0):
      * low PERCLOS (0.01 - 0.12)
      * healthy EAR (0.26 - 0.38)
      * normal blink rate (10 - 22 BPM)
      * brief blinks (120 - 280 ms)
      * max closure < 0.4s
    - Drowsy Class (1):
      * elevated PERCLOS (0.22 - 0.75)
      * depressed EAR (0.12 - 0.23)
      * erratic / sluggish blinks (350 - 900 ms) or staring (<6 BPM) or fluttering (>28 BPM)
      * extended closure (0.8s - 3.5s microsleeps)
    """
    np.random.seed(random_seed)
    
    half = n_samples // 2
    X = []
    y = []

    # 1. Generate Alert Samples (y = 0)
    for _ in range(half):
        perclos = float(np.clip(np.random.beta(1.5, 14.0), 0.01, 0.18))
        mean_ear = float(np.random.normal(0.31, 0.03))
        min_ear = float(np.clip(mean_ear - np.random.uniform(0.08, 0.14), 0.15, 0.28))
        closed_dur = float(perclos * 10.0)
        bpm = float(np.clip(np.random.normal(16.0, 3.5), 8.0, 24.0))
        blink_dur = float(np.random.normal(200.0, 35.0))
        longest_closure = float(np.random.uniform(0.10, 0.35))
        ear_var = float(np.random.uniform(0.001, 0.006))

        X.append([perclos, mean_ear, min_ear, closed_dur, bpm, blink_dur, longest_closure, ear_var])
        y.append(0)

    # 2. Generate Drowsy Samples (y = 1)
    for _ in range(half):
        perclos = float(np.clip(np.random.beta(4.0, 5.0), 0.20, 0.85))
        mean_ear = float(np.random.normal(0.18, 0.035))
        min_ear = float(np.clip(mean_ear - np.random.uniform(0.06, 0.12), 0.04, 0.16))
        closed_dur = float(perclos * 10.0)
        
        # Bimodal blink behavior: either fatigued staring (<6 BPM) or compensatory flutter (>28 BPM)
        if np.random.rand() > 0.4:
            bpm = float(np.clip(np.random.normal(5.0, 2.0), 1.0, 8.0))
        else:
            bpm = float(np.clip(np.random.normal(30.0, 4.0), 25.0, 42.0))

        blink_dur = float(np.random.normal(460.0, 90.0))
        longest_closure = float(np.random.uniform(0.9, 3.8))
        ear_var = float(np.random.uniform(0.008, 0.035))

        X.append([perclos, mean_ear, min_ear, closed_dur, bpm, blink_dur, longest_closure, ear_var])
        y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def train_perclos_model(n_samples=3000, model_type="svm"):
    """
    Trains and evaluates the PERCLOS Machine Learning classifier.
    """
    print("=" * 70)
    print("  TRAINING PERCLOS MACHINE LEARNING CLASSIFIER")
    print("=" * 70)
    print(f"Features: {FEATURE_NAMES}")
    print(f"Sample Count: {n_samples} | Model Algorithm: {model_type.upper()}")

    # 1. Dataset Generation & Splitting
    X, y = generate_perclos_dataset(n_samples=n_samples, random_seed=config.RANDOM_SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.RANDOM_SEED, stratify=y
    )

    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # 2. Build Pipeline (StandardScaler + Classifier)
    if model_type.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=config.RANDOM_SEED)
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    else:  # Default SVM with probability calibration
        clf = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=config.RANDOM_SEED)
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    # 3. Fit Model
    pipeline.fit(X_train, y_train)

    # 4. Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, average=None)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nHeld-Out Test Results:")
    print(f" - Test Accuracy:  {acc * 100:.2f}%")
    print(f" - Drowsy Recall:  {rec[1] * 100:.2f}% (Sensitivity)")
    print(f" - Alert Specificity: {rec[0] * 100:.2f}%")
    print(f" - Macro F1-Score: {macro_f1 * 100:.2f}%")

    # 5. Save Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = config.MODELS_DIR / "perclos_model.joblib"
    joblib.dump(pipeline, model_save_path)

    # Confusion matrix plot
    cm_plot_path = config.MODELS_DIR / "perclos_confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=["Alert", "Drowsy"],
        yticklabels=["Alert", "Drowsy"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("PERCLOS ML Model Confusion Matrix", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted State")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Save Evaluation Report JSON
    eval_report = {
        "status": "COMPLETED",
        "model_type": model_type.upper(),
        "features": FEATURE_NAMES,
        "test_samples": len(y_test),
        "test_accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "drowsy_sensitivity_recall": round(float(rec[1]), 4),
        "alert_specificity_recall": round(float(rec[0]), 4),
        "per_class_metrics": {
            "Alert": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1[0]), 4)},
            "Drowsy": {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1[1]), 4)}
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1])
        },
        "literature_reference": "Kumar et al., 2023 - PERCLOS-Based ML Drowsiness Application (IJRASET)"
    }

    eval_json_path = config.MODELS_DIR / "perclos_evaluation_report.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nModel saved to: {model_save_path}")
    print(f"Report saved to: {eval_json_path}")
    print("=" * 70)

    return pipeline, eval_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "rf"])
    args = parser.parse_args()

    train_perclos_model(n_samples=args.samples, model_type=args.model)
