"""
Lightweight Eye-State Classification Model Training Pipeline (Open vs Closed)
Uses TensorFlow/Keras to train a lightweight, mobile-optimized ConvNet.

Features:
- Configurable dataset directory (dataset/ with train/val/test splits or flat open/closed)
- Image preprocessing: 64x64 resolution, [-1, 1] normalization, data augmentation
- Lightweight architecture (< 150k params) suitable for edge inference
- ModelCheckpoint saving best validation loss
- Generates loss/accuracy curves to backend/models/eye_training_history.png
- Saves model artifact to backend/models/eye_state_model.keras with metadata
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np

# Suppress excessive TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lightweight_eye_model(input_shape=(64, 64, 3)):
    """
    Constructs a lightweight convolutional network for binary eye-state classification.
    0 = Open Eye, 1 = Closed Eye.
    """
    inputs = keras.Input(shape=input_shape, name="eye_input")

    # Data augmentation block active during training
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.05)(x)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification Head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="eye_state_pred")(x)

    model = keras.Model(inputs, outputs, name="EyeStateClassifier_V1")
    return model


def discover_eye_images(dataset_dir: Path) -> dict:
    """
    Discovers image file paths and labels (0 = open, 1 = closed).
    Returns dict with keys: 'train', 'val', 'test', each containing (paths, labels).
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # Check if pre-split structure exists: dataset/train/{open, closed}
    train_open_dir = dataset_dir / "train" / "open"
    train_closed_dir = dataset_dir / "train" / "closed"

    if train_open_dir.exists() and train_closed_dir.exists():
        splits = {}
        for split_name in ["train", "validation", "test"]:
            s_open = dataset_dir / split_name / "open"
            s_closed = dataset_dir / split_name / "closed"
            paths, labels = [], []

            if s_open.exists():
                for f in s_open.iterdir():
                    if f.suffix.lower() in valid_exts:
                        paths.append(str(f))
                        labels.append(0)

            if s_closed.exists():
                for f in s_closed.iterdir():
                    if f.suffix.lower() in valid_exts:
                        paths.append(str(f))
                        labels.append(1)

            key = "val" if split_name == "validation" else split_name
            splits[key] = (paths, labels)

        total = sum(len(p) for p, _ in splits.values())
        if total > 0:
            return splits

    # Check flat structure: dataset/open and dataset/closed
    flat_open_dir = dataset_dir / "open"
    flat_closed_dir = dataset_dir / "closed"
    if flat_open_dir.exists() and flat_closed_dir.exists():
        all_paths, all_labels = [], []
        for f in flat_open_dir.iterdir():
            if f.suffix.lower() in valid_exts:
                all_paths.append(str(f))
                all_labels.append(0)
        for f in flat_closed_dir.iterdir():
            if f.suffix.lower() in valid_exts:
                all_paths.append(str(f))
                all_labels.append(1)

        if not all_paths:
            raise FileNotFoundError(f"No valid image files found inside {dataset_dir}/open or {dataset_dir}/closed.")

        # Split 70% train, 15% val, 15% test
        p_train, p_temp, l_train, l_temp = train_test_split(all_paths, all_labels, test_size=0.30, random_state=42, stratify=all_labels)
        p_val, p_test, l_val, l_test = train_test_split(p_temp, l_temp, test_size=0.50, random_state=42, stratify=l_temp)

        return {
            "train": (p_train, l_train),
            "val": (p_val, l_val),
            "test": (p_test, l_test)
        }

    raise FileNotFoundError(
        f"Eye dataset folder structure not recognized at: {dataset_dir}\n"
        "Expected either:\n"
        "  1) dataset/train/open/, dataset/train/closed/, etc.\n"
        "  2) dataset/open/ and dataset/closed/\n"
        "Run 'python prepare_eye_dataset.py --extract-from-ddd' or download MRL Eye Dataset."
    )


def load_and_preprocess_sample(path_tensor, label_tensor, target_size=(64, 64)):
    """Loads image bytes, decodes, resizes, and normalizes to [-1, 1]."""
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size, method="area")
    # Normalize to [-1, 1]
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img, label_tensor


def create_tf_dataset(paths, labels, batch_size=32, is_training=False, target_size=(64, 64)):
    """Creates an efficient tf.data pipeline."""
    paths_ds = tf.constant(paths)
    labels_ds = tf.constant(labels, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths_ds, labels_ds))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=42)

    ds = ds.map(
        lambda p, l: load_and_preprocess_sample(p, l, target_size=target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_training_history(history, save_path):
    """Generates and saves dual-pane loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#06b6d4", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#10b981", linewidth=2)
    ax1.set_title("Eye Model Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", color="#f59e0b", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", color="#f43f5e", linewidth=2)
    ax2.set_title("Eye Model Binary Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training history saved to: {save_path}")


def train_eye_model(
    dataset_dir=None,
    epochs=15,
    batch_size=32,
    learning_rate=1e-3,
    smoke_test=False
):
    """Executes the complete training workflow."""
    if dataset_dir is None:
        dataset_dir = config.EYE_DATASET_DIR
    dataset_dir = Path(dataset_dir)

    print("=" * 70)
    print("  AIoT EYE-STATE CLASSIFICATION MODEL TRAINING")
    print("=" * 70)
    print(f"Target Dataset Directory: {dataset_dir}")
    print(f"Target Model Artifact: {config.EYE_MODEL_PATH}")

    if not dataset_dir.exists():
        print(f"\n[ERROR] Dataset directory not found at: {dataset_dir}")
        print("Please run: python prepare_eye_dataset.py --extract-from-ddd")
        return {"status": "ERROR", "message": f"Dataset directory not found: {dataset_dir}"}

    set_all_seeds(42)

    try:
        splits = discover_eye_images(dataset_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return {"status": "ERROR", "message": str(e)}

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    if smoke_test:
        print("\n[SMOKE TEST MODE ACTIVATED] Subsampling dataset for quick validation...")
        train_paths, train_labels = train_paths[:60], train_labels[:60]
        val_paths, val_labels = val_paths[:20], val_labels[:20]
        test_paths, test_labels = test_paths[:20], test_labels[:20]
        epochs = min(epochs, 2)
        batch_size = 8

    print(f"\nDataset Split Statistics:")
    print(f" - Train Samples: {len(train_paths)} (Open: {train_labels.count(0)}, Closed: {train_labels.count(1)})")
    print(f" - Val Samples:   {len(val_paths)} (Open: {val_labels.count(0)}, Closed: {val_labels.count(1)})")
    print(f" - Test Samples:  {len(test_paths)} (Open: {test_labels.count(0)}, Closed: {test_labels.count(1)})")

    if len(train_paths) == 0:
        print("[ERROR] Training set is empty.")
        return {"status": "ERROR", "message": "Training set is empty"}

    # Save test split paths for evaluation script
    test_split_info = {
        "test_paths": test_paths,
        "test_labels": test_labels
    }
    test_split_path = config.MODELS_DIR / "eye_test_split.json"
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump(test_split_info, f, indent=2)

    # Class weights for balance
    classes = np.array([0, 1])
    class_weights_arr = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
    class_weights_dict = {0: float(class_weights_arr[0]), 1: float(class_weights_arr[1])}
    print(f"Class Weights: Open (0) = {class_weights_dict[0]:.2f}, Closed (1) = {class_weights_dict[1]:.2f}")

    # Build tf.data pipelines
    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, target_size=config.EYE_IMAGE_SIZE)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False, target_size=config.EYE_IMAGE_SIZE)

    # Instantiate model
    model = build_lightweight_eye_model(input_shape=(*config.EYE_IMAGE_SIZE, 3))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(config.EYE_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )
    duration = round(time.time() - t_start, 2)

    # Save training curves
    history_png = config.MODELS_DIR / "eye_training_history.png"
    plot_training_history(history, history_png)

    # Save training metadata
    metadata = {
        "model_architecture": "Lightweight ConvNet (3 Conv Blocks + GlobalAvgPool)",
        "input_shape": [*config.EYE_IMAGE_SIZE, 3],
        "classes": {"open": 0, "closed": 1},
        "training_mode": "SMOKE_TEST" if smoke_test else "FULL",
        "total_dataset_images": len(train_paths) + len(val_paths) + len(test_paths),
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "batch_size": batch_size,
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": round(min(history.history["val_loss"]), 4),
        "best_val_accuracy": round(max(history.history["val_accuracy"]), 4),
        "training_duration_sec": duration,
        "saved_model_path": str(config.EYE_MODEL_PATH),
        "created_at": datetime.now().isoformat(),
        "model_size_mb": round(os.path.getsize(config.EYE_MODEL_PATH) / (1024 * 1024), 2) if config.EYE_MODEL_PATH.exists() else 0.0
    }

    with open(config.EYE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining completed successfully in {duration}s!")
    print(f"Model saved to: {config.EYE_MODEL_PATH}")
    print(f"Metadata saved to: {config.EYE_METADATA_PATH}")
    return {"status": "SUCCESS", "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIoT Eye State Model Trainer")
    parser.add_argument("--data-dir", type=str, default=str(config.EYE_DATASET_DIR), help="Eye dataset directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 2-epoch smoke test on subset")
    args = parser.parse_args()

    train_eye_model(
        dataset_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test
    )
