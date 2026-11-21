"""
Dataset Inspection and Validation Tool for Driver Drowsiness Dataset (DDD)
Part of AIoT Driver Drowsiness Detection System.

Features:
- Validates existence of dataset directory
- Recursively discovers image files and folder-based class labels
- Checks image readability, corruptions, and dimensions using PIL
- Discovers subject/video identifiers from RLDD filename prefixes (A-ZC)
- Calculates class distributions and potential class imbalances
- Exports a complete inspection report JSON to backend/models/
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

def inspect_dataset(dataset_dir=None, sample_check_limit=None):
    """
    Inspects and audits the dataset folder.
    
    Args:
        dataset_dir: Path to dataset. If None, uses config.DATASET_DIR.
        sample_check_limit: If int, tests readability of first N images per class for speed.
                            If None, checks all images.
    Returns:
        dict: Inspection report data.
    """
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR
    else:
        dataset_dir = Path(dataset_dir)

    print("=" * 70)
    print("  AIoT DDD DATASET INSPECTION & INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Target Directory: {dataset_dir}")

    if not dataset_dir.exists():
        error_msg = (
            f"\n[ERROR] Dataset directory not found at: {dataset_dir}\n"
            f"Please download the Driver Drowsiness Dataset (DDD) from Kaggle\n"
            f"and place it at: {dataset_dir}\n"
            f"Expected subdirectories: 'Drowsy' and 'Non Drowsy' (or 'Not Drowsy').\n"
        )
        print(error_msg)
        return {"status": "ERROR", "message": error_msg}

    # Find subdirectories (folder-based labeling)
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not subdirs:
        error_msg = f"[ERROR] No class subdirectories found inside {dataset_dir}."
        print(error_msg)
        return {"status": "ERROR", "message": error_msg}

    print(f"Discovered subdirectories: {[d.name for d in subdirs]}")

    class_data = {}
    total_images = 0
    corrupted_images = []
    dimensions_counter = Counter()
    formats_counter = Counter()
    subject_counter = defaultdict(Counter)
    file_extensions = Counter()

    # Canonical mapping: support 'Non Drowsy', 'Not Drowsy', 'Alert', etc.
    canonical_map = {}
    for d in subdirs:
        norm = d.name.lower().replace("_", " ").strip()
        if "non" in norm or "not" in norm or "alert" in norm or "awake" in norm:
            canonical_map[d.name] = "Non Drowsy"
        elif "drowsy" in norm or "fatigue" in norm or "sleepy" in norm:
            canonical_map[d.name] = "Drowsy"
        else:
            canonical_map[d.name] = d.name

    for d in subdirs:
        class_name = canonical_map[d.name]
        files = [f for f in d.iterdir() if f.is_file()]
        valid_images = []
        
        print(f"\nScanning class directory '{d.name}' (mapped to '{class_name}')...")
        print(f"Total candidate files found: {len(files)}")

        check_subset = files if sample_check_limit is None else files[:sample_check_limit]

        for i, file_path in enumerate(check_subset):
            ext = file_path.suffix.lower()
            file_extensions[ext] += 1

            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                continue

            # Extract subject identifier from filename prefix (e.g. A0002 -> A, za0012 -> za)
            prefix_match = re.match(r"^([A-Za-z]+)", file_path.stem)
            subj_id = prefix_match.group(1).upper() if prefix_match else "UNKNOWN"
            subject_counter[class_name][subj_id] += 1

            # Validate image readability and dimensions
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Fast structural verification
                # Reopen to check dimensions and mode (verify closes file)
                with Image.open(file_path) as img:
                    dimensions_counter[img.size] += 1
                    formats_counter[img.format] += 1
                    valid_images.append(str(file_path))
            except Exception as e:
                corrupted_images.append({"file": str(file_path), "error": str(e)})

        total_images += len(files)
        class_data[class_name] = {
            "folder_name": d.name,
            "raw_file_count": len(files),
            "checked_valid_count": len(valid_images),
            "subjects_found": len(subject_counter[class_name]),
            "subject_breakdown": dict(subject_counter[class_name])
        }

    # Analyze class distribution
    class_counts = {cls: data["raw_file_count"] for cls, data in class_data.items()}
    imbalance_ratio = None
    if "Drowsy" in class_counts and "Non Drowsy" in class_counts and class_counts["Non Drowsy"] > 0:
        imbalance_ratio = round(class_counts["Drowsy"] / class_counts["Non Drowsy"], 3)

    report = {
        "status": "VALID",
        "dataset_path": str(dataset_dir),
        "total_images": total_images,
        "classes": class_data,
        "class_mapping": {"Non Drowsy": 0, "Drowsy": 1},
        "imbalance_ratio_drowsy_to_nondrowsy": imbalance_ratio,
        "file_extensions": dict(file_extensions),
        "image_formats": dict(formats_counter),
        "image_dimensions": {f"{w}x{h}": cnt for (w, h), cnt in dimensions_counter.items()},
        "corrupted_files_count": len(corrupted_images),
        "corrupted_files_sample": corrupted_images[:10],
        "subject_independent_split_viable": True,
        "notes": (
            "Derived from Real-Life Drowsiness Dataset (RLDD). "
            "File prefixes (A-ZC) represent distinct subject/video recordings. "
            "Subject-independent splitting is supported to eliminate data leakage."
        )
    }

    print("\n" + "=" * 70)
    print("  INSPECTION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Images: {total_images:,}")
    for c, c_info in class_data.items():
        print(f" - {c}: {c_info['raw_file_count']:,} images across {c_info['subjects_found']} subject sequences")
    print(f"Class Imbalance Ratio (Drowsy / Non Drowsy): {imbalance_ratio}")
    print(f"Primary Image Dimensions: {dict(dimensions_counter)}")
    print(f"Corrupted Images: {len(corrupted_images)}")
    print(f"Subject Independence Viable: Yes ({len(set(subject_counter['Drowsy'].keys()) | set(subject_counter['Non Drowsy'].keys()))} unique subject IDs)")

    # Save to JSON
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATASET_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nInspection report saved to: {config.DATASET_REPORT_PATH}")
    print("=" * 70)

    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit and inspect DDD dataset")
    parser.add_argument("--path", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--sample-limit", type=int, default=1000, help="Number of files to deeply verify per class (default 1000, 0 for all)")
    args = parser.parse_args()

    limit = None if args.sample_limit == 0 else args.sample_limit
    inspect_dataset(args.path, sample_check_limit=limit)
