"""
Preprocessing Pipeline for Driver Drowsiness Detection System
Provides unified, deterministic preprocessing for both model training and live webcam inference.
"""

import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224), use_mobilenet_norm=True):
    """
    Standard preprocessing for MobileNetV2 input.
    
    Args:
        image: Can be a file path (str), a numpy array (BGR or RGB), or a PIL Image.
        target_size: Tuple (width, height), default (224, 224).
        use_mobilenet_norm: If True, scales pixel values to [-1, 1] using standard MobileNetV2 formula:
                            (img / 127.5) - 1.0. If False, scales to [0, 1].
    
    Returns:
        np.ndarray of shape (1, height, width, 3), float32.
    """
    # 1. Handle file path input
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Unable to read image file at: {image}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 2. Handle PIL Image
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    # 3. Handle numpy array
    elif isinstance(image, np.ndarray):
        # If OpenCV frame (BGR assumed if standard 3-channel uint8), ensure RGB
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Caller can pass BGR or RGB; for safety in inference, caller passes BGR frame
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Resize to model input dimensions using high-quality interpolation
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Convert to float32
    processed = resized.astype(np.float32)

    # Normalize: MobileNetV2 expectation is [-1, 1]
    if use_mobilenet_norm:
        processed = (processed / 127.5) - 1.0
    else:
        processed = processed / 255.0

    # Expand dims to batch shape: (1, 224, 224, 3)
    batched = np.expand_dims(processed, axis=0)
    return batched

def extract_face_crop(frame_bgr, bounding_box, margin=0.15, target_size=(224, 224)):
    """
    Crops a detected face from a video frame with a margin to match the DDD dataset face framing.
    
    Args:
        frame_bgr: Full BGR camera frame
        bounding_box: (x, y, w, h)
        margin: Proportion of padding around bounding box
        target_size: (w, h)
        
    Returns:
        face_crop_bgr: Cropped and resized face image in BGR
    """
    h_frame, w_frame = frame_bgr.shape[:2]
    x, y, w, h = bounding_box

    # Add margin around face
    x_margin = int(w * margin)
    y_margin = int(h * margin)

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(w_frame, x + w + x_margin)
    y2 = min(h_frame, y + h + y_margin)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        face_crop = frame_bgr

    resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    return resized_face, (x1, y1, x2 - x1, y2 - y1)
