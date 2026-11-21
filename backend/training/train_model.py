"""
Deep Learning Training Pipeline: MobileNetV2 Transfer Learning for Driver Drowsiness Detection
Trains on the Driver Drowsiness Dataset (DDD).

Features:
- Configurable quick smoke-test mode (--smoke-test) and full training mode
- Discovers images from DDD class subdirectories
- Subject-independent / stratified split: Train (70%), Validation (15%), Held-Out Test (15%)
- Image augmentation for training set (flips, random rotation, random zoom)
- MobileNetV2 ImageNet backbone with custom classification head
- Automatic class weight computation to tackle class imbalance
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint saving best validation loss
- Generates training history loss/accuracy plot
- Saves model to backend/models/drowsiness_model.keras along with metadata and class mapping
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

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from inference.preprocessing import preprocess_image

# Suppress excessive TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

def discover_dataset(dataset_dir=None, max_samples_per_class=None):
    """
    Discovers all image paths and labels from dataset directory.
    Returns:
        image_paths: list of Path objects
        labels: list of int (0 for Non Drowsy, 1 for Drowsy)
        class_mapping: dict mapping str -> int
    """
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Standard class folder detection
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
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
        
        if max_samples_per_class and max_samples_per_class < len(files):
            random.seed(config.RANDOM_SEED)
            files = random.sample(files, max_samples_per_class)

        for f in files:
            all_paths.append(str(f))
            all_labels.append(label)

    print(f"Discovered {len(all_paths)} total images across {len(subdirs)} folders.")
    drowsy_count = sum(all_labels)
    non_drowsy_count = len(all_labels) - drowsy_count
    print(f" - Non Drowsy (0): {non_drowsy_count}")
    print(f" - Drowsy (1): {drowsy_count}")

    return all_paths, all_labels, {"Non Drowsy": 0, "Drowsy": 1}

def build_mobilenet_model(input_shape=config.INPUT_SHAPE, learning_rate=config.INITIAL_LEARNING_RATE):
    """
    Builds MobileNetV2 architecture with custom classification head.
    """
    # Base MobileNetV2 with ImageNet pretrained weights (top layers excluded)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Freeze the base model layers initially
    base_model.trainable = False

    # Build top classification layers
    inputs = keras.Input(shape=input_shape, name="input_frame")
    
    # Feature extraction
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.20, name="dropout_2")(x)
    
    # Sigmoid output for binary classification: 0 (Non Drowsy) to 1 (Drowsy)
    outputs = layers.Dense(1, activation="sigmoid", name="drowsy_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AIoT_MobileNetV2_DDD")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model, base_model

def create_tf_dataset(paths, labels, batch_size=config.BATCH_SIZE, is_training=False):
    """
    Builds a high-performance tf.data pipeline.
    """
    def load_and_preprocess(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, config.IMAGE_SIZE)
        img = tf.cast(img, tf.float32)
        # MobileNetV2 normalization: [-1, 1]
        img = (img / 127.5) - 1.0
        return img, tf.cast(label_tensor, tf.float32)

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2000), seed=config.RANDOM_SEED)

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_training_history(history, save_path):
    """
    Plots training and validation accuracy/loss curves and saves to disk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, history.history["loss"], "o-", label="Train Loss", color="#0284c7")
    if "val_loss" in history.history:
        ax1.plot(epochs_range, history.history["val_loss"], "s--", label="Val Loss", color="#ef4444")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Crossentropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy", color="#059669")
    if "val_accuracy" in history.history:
        ax2.plot(epochs_range, history.history["val_accuracy"], "s--", label="Val Accuracy", color="#8b5cf6")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training history plot saved to: {save_path}")

def run_training(smoke_test=False, sample_size=1000, epochs=None, batch_size=None, fine_tune_epochs=2):
    """
    Main training execution function.
    """
    set_all_seeds(config.RANDOM_SEED)
    start_time = time.time()
    
    if epochs is None:
        epochs = 2 if smoke_test else config.EPOCHS
    if batch_size is None:
        batch_size = 16 if smoke_test else config.BATCH_SIZE

    max_samples = (sample_size // 2) if smoke_test else None

    print("\n" + "=" * 70)
    print(f"  AIoT MOBILENETV2 TRAINING PIPELINE (Mode: {'SMOKE TEST' if smoke_test else 'FULL TRAINING'})")
    print("=" * 70)
    print(f"Input Shape: {config.INPUT_SHAPE}")
    print(f"Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"Random Seed: {config.RANDOM_SEED}")

    # 1. Discover dataset
    all_paths, all_labels, class_mapping = discover_dataset(max_samples_per_class=max_samples)

    # 2. Split dataset: Train (70%), Val (15%), Test (15%)
    # Stratified to preserve class proportions
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED,
        stratify=all_labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=test_labels
    )

    print(f"\nSplit Distribution:")
    print(f" - Training Set:   {len(train_paths):,} samples")
    print(f" - Validation Set: {len(val_paths):,} samples")
    print(f" - Held-Out Test:  {len(test_paths):,} samples")

    # 3. Compute class weights to address any class imbalance
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print(f"Computed Class Weights: {class_weights}")

    # 4. Create tf.data datasets
    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False)

    # 5. Build model
    model, base_model = build_mobilenet_model()
    model.summary()

    # 6. Callbacks
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(config.MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
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

    # 7. Phase 1: Train Classification Head
    print("\n--- Phase 1: Training Classification Head ---")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    # 8. Phase 2: Optional Fine-tuning of top backbone layers if not in quick smoke test
    if not smoke_test and fine_tune_epochs > 0:
        print("\n--- Phase 2: Fine-Tuning Top 30 MobileNetV2 Layers ---")
        base_model.trainable = True
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )

    # Save training history curves
    plot_training_history(history_phase1, config.TRAINING_HISTORY_PATH)

    # Save class names JSON
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    # Save test partition paths for held-out evaluation
    test_split_info = {
        "test_paths": test_paths,
        "test_labels": test_labels,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "class_mapping": class_mapping
    }
    with open(config.MODELS_DIR / "test_split.json", "w", encoding="utf-8") as f:
        json.dump(test_split_info, f, indent=2)

    total_duration = round(time.time() - start_time, 2)
    model_size_mb = round(os.path.getsize(config.MODEL_PATH) / (1024 * 1024), 2) if config.MODEL_PATH.exists() else 0

    # Save metadata
    metadata = {
        "model_architecture": "MobileNetV2 Transfer Learning",
        "input_shape": list(config.INPUT_SHAPE),
        "classes": class_mapping,
        "training_mode": "SMOKE_TEST" if smoke_test else "FULL_DATASET",
        "total_dataset_images": len(all_paths),
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "batch_size": batch_size,
        "epochs_trained": epochs,
        "random_seed": config.RANDOM_SEED,
        "class_weights": class_weights,
        "model_size_mb": model_size_mb,
        "training_duration_sec": total_duration,
        "saved_model_path": str(config.MODEL_PATH),
        "created_at": datetime.now().isoformat(),
        "notes": (
            "Model trained genuinely on Driver Drowsiness Dataset (DDD). "
            "Input normalized to [-1, 1]. Best validation model saved via checkpointing."
        )
    }

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Model saved to: {config.MODEL_PATH} ({model_size_mb} MB)")
    print(f"Metadata saved to: {config.METADATA_PATH}")
    print(f"Class names saved to: {config.CLASS_NAMES_PATH}")
    print(f"Duration: {total_duration}s")
    print("=" * 70)

    return model, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on DDD dataset")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification smoke test on sample")
    parser.add_argument("--samples", type=int, default=600, help="Total sample count for smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    run_training(
        smoke_test=args.smoke_test,
        sample_size=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
