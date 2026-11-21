# ==============================================================================
# RETRIEVE SAVED MODEL FROM GOOGLE DRIVE (NO RE-TRAINING NEEDED!)
# Run this anytime in a fresh Colab notebook to generate new plots, tables, or tests.
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Mount Drive & Dependencies
# ------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, log_loss,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ------------------------------------------------------------------------------
# STEP 2: Load Your Saved Model Directly from Google Drive
# ------------------------------------------------------------------------------
# Default path where Cell 10 saved your model:
MODEL_DRIVE_PATH = "/content/drive/MyDrive/AIoT_Drowsiness_Results/drowsiness_model.keras"

# Check if model exists
if not os.path.exists(MODEL_DRIVE_PATH):
    # Search drive automatically if in another folder
    candidates = glob.glob("/content/drive/MyDrive/**/drowsiness_model.keras", recursive=True)
    if candidates:
        MODEL_DRIVE_PATH = candidates[0]
    else:
        raise FileNotFoundError(f"Model file not found in Google Drive! Please check Drive path.")

print(f"📦 Loading pre-trained model from: {MODEL_DRIVE_PATH} ...")
model = keras.models.load_model(MODEL_DRIVE_PATH)
print("✅ Model loaded successfully into memory! Ready to evaluate.")
model.summary()

# ------------------------------------------------------------------------------
# STEP 3: Now you can generate ANY result without training!
# (Examples below)
# ------------------------------------------------------------------------------

# Example A: Test on any single image from Drive or web
def predict_single_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0  # Normalize
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    prob = float(model.predict(img)[0][0])
    label = "Drowsy" if prob >= 0.50 else "Non Drowsy"
    conf = prob if prob >= 0.50 else (1.0 - prob)
    return label, conf, prob

# Example B: Generate custom Precision-Recall Curve or Custom Heatmaps
# Example C: Re-run test metrics on new datasets or unseen videos
