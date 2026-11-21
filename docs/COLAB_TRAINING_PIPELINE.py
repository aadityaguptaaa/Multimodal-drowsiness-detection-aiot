# ==============================================================================
# AIoT DRIVER DROWSINESS DETECTION - GOOGLE COLAB COMPLETE TRAINING PIPELINE
# Architecture: MobileNetV2 Transfer Learning + Top 30 Layer Fine-Tuning
# Dataset: Driver Drowsiness Dataset (DDD)
# Author: AIoT Drowsiness Detection Project Team
# ==============================================================================

# ==============================================================================
# CELL 1: HARDWARE & GPU VERIFICATION
# ==============================================================================
# In Google Colab: Go to Runtime -> Change runtime type -> Select T4 GPU or A100 GPU
import os
import sys
import time
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)

print(f"TensorFlow Version : {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU Detected      : {gpus[0].name}")
    !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else:
    print("⚠️ WARNING: No GPU detected! Running on CPU. (Go to Runtime -> Change runtime type -> T4 GPU)")

# Set deterministic random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================================================================
# CELL 2: DIRECT DATASET UPLOAD & EXTRACTION (NO GOOGLE DRIVE REQUIRED)
# ==============================================================================
# In Google Colab left sidebar: Click the "Folder" icon 📁 -> Click "Upload" ⬆️
# Upload your zipped dataset (e.g., Driver_Drowsiness_Dataset.zip or dataset.zip)
# directly into /content/
#
# Alternatively, you can run:
# from google.colab import files
# uploaded = files.upload()

import glob
from pathlib import Path

LOCAL_EXTRACT_DIR = "/content/dataset"
os.makedirs(LOCAL_EXTRACT_DIR, exist_ok=True)

# 1. Search for any uploaded .zip files in /content/
zip_candidates = glob.glob("/content/*.zip")

if zip_candidates:
    chosen_zip = zip_candidates[0]
    print(f"📦 Found uploaded zip file: {chosen_zip}")
    print(f"Extracting to {LOCAL_EXTRACT_DIR}...")
    !unzip -q "{chosen_zip}" -d "{LOCAL_EXTRACT_DIR}"
    print("✅ Extraction complete!")
else:
    print("ℹ️ No .zip file found in /content/. Checking if dataset folder was uploaded unzipped...")

# 2. Auto-detect dataset root folder containing 'Drowsy' and 'Non Drowsy'
possible_dirs = [
    LOCAL_EXTRACT_DIR,
    "/content",
    "/content/Driver Drowsiness Dataset (DDD)",
    "/content/dataset/Driver Drowsiness Dataset (DDD)",
    "/content/dataset/DDD"
]

DATASET_ROOT = None
for p in possible_dirs:
    p_path = Path(p)
    if p_path.exists():
        for candidate in [p_path] + [d for d in p_path.glob("**/*") if d.is_dir()]:
            subdirs = [d.name.lower() for d in candidate.iterdir() if d.is_dir()]
            if any("drowsy" in s for s in subdirs) and any("non" in s or "not" in s or "alert" in s for s in subdirs):
                DATASET_ROOT = candidate
                break
    if DATASET_ROOT is not None:
        break

if DATASET_ROOT is None:
    print("\n❌ Could not automatically detect the 'Drowsy' and 'Non Drowsy' folders.")
    print("👉 How to fix in 10 seconds:")
    print("1. Click the Folder icon (📁) on the left sidebar of Google Colab.")
    print("2. Upload your 'Driver_Drowsiness_Dataset.zip'.")
    print("3. Re-run this cell!")
    raise FileNotFoundError("Dataset not found in /content/. Please upload your zip file into Colab.")

print(f"✅ Active Dataset Root Detected: {DATASET_ROOT}")


# ==============================================================================
# CELL 3: DATASET DISCOVERY, AUDIT & PREPROCESSING
# ==============================================================================
CLASS_NAMES = ["Non Drowsy", "Drowsy"]
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 5
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

# Scan subdirectories
subdirs = [d for d in DATASET_ROOT.iterdir() if d.is_dir()]
class_map = {}
for d in subdirs:
    norm = d.name.lower().replace("_", " ").strip()
    if "non" in norm or "not" in norm or "alert" in norm or "awake" in norm:
        class_map[d.name] = 0  # Non Drowsy
    else:
        class_map[d.name] = 1  # Drowsy

all_paths = []
all_labels = []
valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

for d in subdirs:
    label = class_map[d.name]
    files = [f for f in d.iterdir() if f.suffix.lower() in valid_exts]
    for f in files:
        all_paths.append(str(f))
        all_labels.append(label)

print("\n" + "=" * 65)
print(f"📊 DATASET AUDIT SUMMARY:")
print("=" * 65)
print(f"Total Discovered Images : {len(all_paths):,}")
drowsy_cnt = sum(all_labels)
non_drowsy_cnt = len(all_labels) - drowsy_cnt
print(f" - Class 0 (Non Drowsy) : {non_drowsy_cnt:,} ({non_drowsy_cnt/len(all_paths)*100:.1f}%)")
print(f" - Class 1 (Drowsy)     : {drowsy_cnt:,} ({drowsy_cnt/len(all_paths)*100:.1f}%)")

# Stratified Split: 70% Train, 15% Validation, 15% Held-Out Test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_paths, test_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=test_labels
)

print(f"\n📁 Stratified Split Distribution:")
print(f" - Training Set   (70%) : {len(train_paths):,} samples")
print(f" - Validation Set (15%) : {len(val_paths):,} samples")
print(f" - Held-Out Test  (15%) : {len(test_paths):,} samples")

# Compute class weights to handle any minor imbalance
cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print(f"⚖️ Computed Class Weights: {class_weights}")


# ==============================================================================
# CELL 4: HIGH-THROUGHPUT TF.DATA PIPELINE (DATA AUGMENTATION & NORMALIZATION)
# ==============================================================================
def load_and_preprocess(path_tensor, label_tensor):
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    # MobileNetV2 normalization: scale pixel range [0, 255] to [-1.0, 1.0]
    img = (img / 127.5) - 1.0
    return img, tf.cast(label_tensor, tf.float32)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    return image, label

def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_paths, train_labels, is_training=True)
val_ds   = create_dataset(val_paths, val_labels, is_training=False)
test_ds  = create_dataset(test_paths, test_labels, is_training=False)
print("✅ tf.data Input Pipelines built successfully with parallel prefetch.")


# ==============================================================================
# CELL 5: BUILD MOBILENETV2 ARCHITECTURE WITH CUSTOM CLASSIFICATION HEAD
# ==============================================================================
def build_model():
    # Base MobileNetV2 with pre-trained ImageNet weights, top 1000-class layer excluded
    base_model = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    # Freeze base feature extractor initially to prevent catastrophic forgetting
    base_model.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE, name="input_face")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_fatigue_features")(x)
    x = layers.Dropout(0.20, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="drowsy_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AIoT_MobileNetV2_Drowsiness")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model, base_model

model, base_model = build_model()
model.summary()


# ==============================================================================
# CELL 6: TWO-PHASE TRAINING (FEATURE EXTRACTION + TOP-30 LAYER FINE-TUNING)
# ==============================================================================
MODEL_SAVE_PATH = "/content/drowsiness_model.keras"

cb_checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)
cb_earlystop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)
cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("\n" + "=" * 65)
print("🚀 PHASE 1: TRAINING CUSTOM CLASSIFICATION HEAD (Base Frozen)")
print("=" * 65)
t0 = time.time()
history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[cb_checkpoint, cb_earlystop, cb_reduce_lr]
)

print("\n" + "=" * 65)
print("🚀 PHASE 2: FINE-TUNING TOP 30 MOBILENETV2 BACKBONE LAYERS")
print("=" * 65)
# Unfreeze top 30 layers for task-specific facial fatigue specialization
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Base Model Total Layers: {len(base_model.layers)}")
print(f"Trainable Layers for Fine-Tuning: {sum([1 for l in model.layers if l.trainable])}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
)

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[cb_checkpoint, cb_earlystop]
)
total_train_time = time.time() - t0
print(f"\n✅ Total Training Completed in {total_train_time/60:.2f} minutes!")


# ==============================================================================
# CELL 7: TRAINING & VALIDATION LOSS / ACCURACY CURVES
# ==============================================================================
acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
loss = history_p1.history['loss'] + history_p2.history['loss']
val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']
epochs_range = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(epochs_range, loss, "o-", label="Training Loss", color="#dc2626")
ax1.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#2563eb")
ax1.axvline(x=len(history_p1.history['loss']), color="gray", linestyle=":", label="Phase 2 Fine-Tune Start")
ax1.set_title("Binary Cross-Entropy Loss", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#059669")
ax2.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#7c3aed")
ax2.axvline(x=len(history_p1.history['accuracy']), color="gray", linestyle=":", label="Phase 2 Fine-Tune Start")
ax2.set_title("Classification Accuracy", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("/content/training_curves.png", dpi=300)
plt.show()


# ==============================================================================
# CELL 8: RIGOROUS EVALUATION ON HELD-OUT TEST DATASET (15%)
# ==============================================================================
print("\n" + "=" * 65)
print("🔬 EVALUATING BEST CHECKPOINTED MODEL ON HELD-OUT TEST SET...")
print("=" * 65)

# Load best saved weights
best_model = keras.models.load_model(MODEL_SAVE_PATH)

# Predict probabilities on test dataset
y_true = np.array(test_labels)
y_pred_probs = []

for batch_imgs, _ in test_ds:
    preds = best_model(batch_imgs, training=False)
    y_pred_probs.extend(preds.numpy().flatten())

y_pred_probs = np.array(y_pred_probs)
y_pred_classes = (y_pred_probs >= 0.50).astype(int)

# Calculate Core Metrics
acc = accuracy_score(y_true, y_pred_classes)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average="macro")
cm = confusion_matrix(y_true, y_pred_classes)

print(f"\n🎯 FINAL TEST RESULTS ON {len(y_true):,} UNSEEN SAMPLES:")
print(f" - Test Accuracy    : {acc * 100:.2f}% ({acc:.4f})")
print(f" - Macro Precision  : {prec:.4f}")
print(f" - Macro Recall     : {rec:.4f}")
print(f" - Macro F1-Score   : {f1:.4f}")

# Full Classification Report
print("\n" + "=" * 65)
print("📄 DETAILED CLASSIFICATION REPORT:")
print("=" * 65)
report_str = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, digits=4)
print(report_str)


# ==============================================================================
# CELL 9: HIGH-RESOLUTION CONFUSION MATRIX HEATMAP & ROC CURVE
# ==============================================================================
fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix Heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax_cm
)
ax_cm.set_title("Held-Out Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
ax_cm.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
ax_cm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

ax_roc.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"MobileNetV2 ROC (AUC = {roc_auc:.4f})")
ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=10)
ax_roc.grid(True, linestyle="--", alpha=0.5)
ax_roc.legend(loc="lower right")

plt.tight_layout()
plt.savefig("/content/confusion_matrix_and_roc.png", dpi=300)
plt.show()

# Safety Failure Analysis
tn, fp, fn, tp = cm.ravel()
print("\n" + "=" * 65)
print("🛡️ AUTOMOTIVE SAFETY FAILURE MODE ANALYSIS:")
print("=" * 65)
print(f" - True Positives  (Drowsy Correctly Detected)  : {tp:,}")
print(f" - True Negatives  (Alert Drivers Confirmed)    : {tn:,}")
print(f" - False Positives (Nuisance Alarms)            : {fp:,} ({fp/len(y_true)*100:.2f}%)")
print(f" - False Negatives (Drowsiness Missed - HAZARD) : {fn:,} ({fn/len(y_true)*100:.2f}%)")


# ==============================================================================
# CELL 10: INFERENCE SPEED & LATENCY BENCHMARK (MS/FRAME & FPS)
# ==============================================================================
print("\n" + "=" * 65)
print("⚡ INFERENCE SPEED & LATENCY BENCHMARK:")
print("=" * 65)
dummy = np.random.uniform(-1.0, 1.0, size=(1, *INPUT_SHAPE)).astype(np.float32)

# Warmup iterations
for _ in range(20):
    _ = best_model(dummy, training=False)

latencies = []
for _ in range(100):
    t_start = time.perf_counter()
    _ = best_model(dummy, training=False)
    latencies.append((time.perf_counter() - t_start) * 1000.0)

avg_lat = np.mean(latencies)
std_lat = np.std(latencies)
fps = 1000.0 / avg_lat

print(f" - Average Latency per Frame : {avg_lat:.2f} ms (+/- {std_lat:.2f} ms)")
print(f" - Inference Throughput       : {fps:.1f} FPS")
print(f" - Model File Size on Disk   : {os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024):.2f} MB")


# ==============================================================================
# CELL 11: DIRECT LOCAL DOWNLOAD (NO GOOGLE DRIVE REQUIRED)
# ==============================================================================
# Downloads the trained model directly to your computer's "Downloads" folder!
from google.colab import files

print("📥 Starting direct download to your computer...")
files.download(MODEL_SAVE_PATH)
files.download("/content/training_curves.png")
files.download("/content/confusion_matrix_and_roc.png")
print("✅ Check your browser's Downloads folder for 'drowsiness_model.keras'!")
