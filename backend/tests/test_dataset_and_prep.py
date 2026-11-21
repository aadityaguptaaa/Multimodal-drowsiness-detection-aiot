"""
Automated Unit Tests for Dataset Discovery, Preprocessing, and Model Loading
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from inference.preprocessing import preprocess_image, extract_face_crop
from training.inspect_dataset import inspect_dataset

def test_dataset_discovery_and_structure():
    """Validates that DDD dataset exists and has expected class subdirectories."""
    assert config.DATASET_DIR.exists(), f"Dataset directory missing: {config.DATASET_DIR}"
    subdirs = [d.name for d in config.DATASET_DIR.iterdir() if d.is_dir()]
    assert "Drowsy" in subdirs
    assert any("non" in s.lower() or "not" in s.lower() for s in subdirs)

def test_dataset_inspection_report():
    """Tests that inspect_dataset runs and reports valid non-empty class counts."""
    report = inspect_dataset(sample_check_limit=10)
    assert report["status"] == "VALID"
    assert report["total_images"] > 0
    assert "Drowsy" in report["classes"]

def test_preprocessing_output_shape_and_range():
    """Tests that preprocessing produces (1, 224, 224, 3) float32 in [-1, 1]."""
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_frame, target_size=(224, 224), use_mobilenet_norm=True)
    
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.05 and tensor.max() <= 1.05

def test_face_crop_extraction():
    """Tests face crop extraction from bounding box."""
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    crop, bbox = extract_face_crop(dummy_frame, (50, 50, 100, 100), target_size=(224, 224))
    assert crop.shape == (224, 224, 3)
