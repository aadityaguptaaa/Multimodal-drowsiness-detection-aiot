"""
Eye-Image Dataset Setup & Preparation Utility
Manages acquisition, folder structuring, and eye-region crop extraction
for training the Eye-State Classification Model (Open vs Closed).

Features:
- Validates existence of dataset directory (dataset/)
- Supports MRL Eye Dataset / CEW Dataset folder layout
- Provides automated eye-crop extraction from existing face datasets (e.g., DDD)
- Generates standardized train/validation/test splits
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from inference.face_detector import UnifiedFaceDetector

MRL_DOWNLOAD_INSTRUCTIONS = """
==============================================================================
INSTRUCTIONS FOR ACQUIRING THE EYE STATE DATASET:
==============================================================================
To train on a widely-used, licensed research eye dataset:

Option 1: MRL Eye Dataset (Recommended - 84,898 images, CC BY 4.0)
  1. Download from Kaggle: https://www.kaggle.com/datasets/tauhidul/mrl-eye-dataset
  2. Unzip into: dataset/
     Directory structure:
     dataset/
       ├── train/
       │   ├── open/
       │   └── closed/
       ├── validation/
       │   ├── open/
       │   └── closed/
       └── test/
           ├── open/
           └── closed/

Option 2: Automated Eye-Region Extraction from local Driver Drowsiness Dataset (DDD)
  Run this tool to automatically detect and crop left and right eyes:
  python prepare_eye_dataset.py --extract-from-ddd --max-samples 1000

==============================================================================
"""


def setup_dataset_structure(target_dir: Path) -> dict:
    """Creates the standard directory tree for training, validation, and testing."""
    splits = ["train", "validation", "test"]
    classes = ["open", "closed"]
    dirs = {}
    for s in splits:
        for c in classes:
            folder = target_dir / s / c
            folder.mkdir(parents=True, exist_ok=True)
            dirs[f"{s}_{c}"] = folder
    return dirs


def check_dataset_status(dataset_dir: Path) -> dict:
    """Inspects target directory and counts available eye images."""
    if not dataset_dir.exists():
        return {"exists": False, "total": 0, "message": "Dataset directory does not exist."}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    counts = {"train_open": 0, "train_closed": 0, "val_open": 0, "val_closed": 0, "test_open": 0, "test_closed": 0, "flat_open": 0, "flat_closed": 0}

    # Check split layout
    for s in ["train", "validation", "test"]:
        for c in ["open", "closed"]:
            folder = dataset_dir / s / c
            if folder.exists():
                key = f"{'val' if s == 'validation' else s}_{c}"
                counts[key] = len([f for f in folder.iterdir() if f.suffix.lower() in valid_exts])

    # Check flat layout (dataset/open, dataset/closed)
    for c in ["open", "closed"]:
        folder = dataset_dir / c
        if folder.exists():
            counts[f"flat_{c}"] = len([f for f in folder.iterdir() if f.suffix.lower() in valid_exts])

    total_split = sum(counts[k] for k in ["train_open", "train_closed", "val_open", "val_closed", "test_open", "test_closed"])
    total_flat = counts["flat_open"] + counts["flat_closed"]

    return {
        "exists": True,
        "is_split": total_split > 0,
        "is_flat": total_flat > 0,
        "total_images": total_split if total_split > 0 else total_flat,
        "counts": counts
    }


def extract_eyes_from_face(face_bgr: np.ndarray, landmarks_pts: Optional[np.ndarray], target_size=(64, 64)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts left and right eye image crops using landmark coordinates or eye proportions.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None, None

    h, w = face_bgr.shape[:2]

    if landmarks_pts is not None and len(landmarks_pts) >= 468:
        # Landmark indices for MediaPipe eye regions:
        # Left eye: ~33, 133, 159, 145; Right eye: ~362, 263, 386, 374
        left_indices = [33, 133, 159, 145, 153, 154, 155, 144]
        right_indices = [362, 263, 386, 374, 380, 381, 382, 373]

        def crop_eye_landmarks(indices):
            pts = landmarks_pts[indices]
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            eye_w = max_x - min_x
            eye_h = max_y - min_y
            pad_x = int(eye_w * 0.3)
            pad_y = int(eye_h * 0.4)
            x1 = max(0, int(min_x - pad_x))
            y1 = max(0, int(min_y - pad_y))
            x2 = min(w, int(max_x + pad_x))
            y2 = min(h, int(max_y + pad_y))
            crop = face_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            return None

        left_crop = crop_eye_landmarks(left_indices)
        right_crop = crop_eye_landmarks(right_indices)
        return left_crop, right_crop
    else:
        # Anthropometric eye zone fallback (eyes are typically in upper 25%-55% band of face)
        y1 = int(h * 0.25)
        y2 = int(h * 0.55)
        x_mid = w // 2

        left_zone = face_bgr[y1:y2, int(w * 0.1):int(x_mid - w * 0.05)]
        right_zone = face_bgr[y1:y2, int(x_mid + w * 0.05):int(w * 0.9)]

        left_crop = cv2.resize(left_zone, target_size, interpolation=cv2.INTER_AREA) if left_zone.size > 0 else None
        right_crop = cv2.resize(right_zone, target_size, interpolation=cv2.INTER_AREA) if right_zone.size > 0 else None
        return left_crop, right_crop


def extract_eye_dataset_from_ddd(ddd_dir: Path, output_dir: Path, max_samples_per_class: int = 500):
    """
    Extracts genuine eye-region crops from local DDD face images.
    - Drowsy faces -> eye crops categorized as closed (or high EAR fatigue)
    - Non Drowsy faces -> eye crops categorized as open
    """
    print(f"Extracting eye crops from DDD: {ddd_dir} -> {output_dir}")
    dirs = setup_dataset_structure(output_dir)
    detector = UnifiedFaceDetector()

    valid_exts = {".jpg", ".png", ".jpeg"}
    random.seed(42)

    categories = [
        ("Non Drowsy", "open"),
        ("Drowsy", "closed")
    ]

    for ddd_folder, eye_class in categories:
        src_path = ddd_dir / ddd_folder
        if not src_path.exists():
            print(f"[Warning] Folder {src_path} not found.")
            continue

        files = [f for f in src_path.iterdir() if f.suffix.lower() in valid_exts]
        if len(files) > max_samples_per_class:
            files = random.sample(files, max_samples_per_class)

        print(f"Processing {len(files)} face images from {ddd_folder} -> '{eye_class}' eyes...")

        saved_crops = []
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue

            detection = detector.detect(img)
            face_img = img if detection["face_crop_bgr"] is None else detection["face_crop_bgr"]
            l_eye, r_eye = extract_eyes_from_face(face_img, detection["landmarks"])

            for idx_eye, eye_crop in enumerate([l_eye, r_eye]):
                if eye_crop is not None:
                    saved_crops.append(eye_crop)

        # Distribute into train (70%), validation (15%), test (15%)
        random.shuffle(saved_crops)
        n_total = len(saved_crops)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_set = saved_crops[:n_train]
        val_set = saved_crops[n_train:n_train + n_val]
        test_set = saved_crops[n_train + n_val:]

        for subset_name, dataset in [("train", train_set), ("validation", val_set), ("test", test_set)]:
            target_folder = dirs[f"{subset_name}_{eye_class}"]
            for idx, crop in enumerate(dataset):
                filename = target_folder / f"{eye_class}_{idx:05d}.png"
                cv2.imwrite(str(filename), crop)

        print(f"  Saved {eye_class}: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test crops.")

    print("\nEye crop extraction completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIoT Eye Dataset Preparation Tool")
    parser.add_argument("--data-dir", type=str, default=str(config.EYE_DATASET_DIR), help="Target dataset directory")
    parser.add_argument("--extract-from-ddd", action="store_true", help="Extract genuine eye crops from DDD")
    parser.add_argument("--max-samples", type=int, default=400, help="Max face samples per class to extract")
    args = parser.parse_args()

    target_path = Path(args.data_dir)

    if args.extract_from_ddd:
        ddd_path = config.DATASET_DIR
        if not ddd_path.exists():
            print(f"[Error] Driver Drowsiness Dataset not found at: {ddd_path}")
            sys.exit(1)
        extract_eye_dataset_from_ddd(ddd_path, target_path, max_samples_per_class=args.max_samples)
    else:
        status = check_dataset_status(target_path)
        print("=" * 70)
        print("  AIoT EYE DATASET STATUS REPORT")
        print("=" * 70)
        print(f"Directory: {target_path}")
        print(f"Exists: {status['exists']}")
        print(f"Total Images: {status['total_images']}")
        print(f"Counts: {status.get('counts', {})}")
        if status["total_images"] == 0:
            print(MRL_DOWNLOAD_INSTRUCTIONS)
        else:
            print("\nDataset ready for training.")
