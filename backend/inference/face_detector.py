"""
Face Detection and Facial Landmark Extraction Pipeline
Integrates:
1. MediaPipe Tasks FaceLandmarker (High accuracy 478 landmarks for EAR & eye analysis)
2. OpenCV Haar Cascade (Fast fallback for bounding boxes if landmarks fail or camera glare obscures landmarks)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

class UnifiedFaceDetector:
    """
    Robust face detection and landmark extraction.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.landmarker = None
        self.use_mediapipe = False
        self.haar_face = None
        self.haar_eye = None

        # 1. Initialize Haar Cascades for reliable fallback
        cascades_dir = config.MODELS_DIR / "cascades"
        face_xml = cascades_dir / "haarcascade_frontalface_default.xml"
        eye_xml = cascades_dir / "haarcascade_eye.xml"
        if face_xml.exists():
            self.haar_face = cv2.CascadeClassifier(str(face_xml))
        if eye_xml.exists():
            self.haar_eye = cv2.CascadeClassifier(str(eye_xml))

        # 2. Initialize MediaPipe FaceLandmarker
        task_model_path = config.MODELS_DIR / "face_landmarker.task"
        if task_model_path.exists():
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                base_options = python.BaseOptions(model_asset_path=str(task_model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
                print("[FaceDetector] MediaPipe FaceLandmarker successfully initialized.")
            except Exception as e:
                print(f"[FaceDetector Warning] MediaPipe landmarker initialization failed: {e}. Falling back to Haar.")
                self.use_mediapipe = False

    def detect(self, frame_bgr):
        """
        Processes a single BGR camera frame.
        
        Returns:
            dict containing:
            - face_found: bool
            - bbox: (x, y, w, h)
            - face_crop_bgr: np.ndarray (224x224)
            - landmarks: dict or list of 2D/3D points (if mediapipe succeeded)
            - detector_type: 'MEDIAPIPE' or 'HAAR_CASCADE'
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return {
                "face_found": False,
                "bbox": None,
                "face_crop_bgr": None,
                "landmarks": None,
                "detector_type": "NONE"
            }

        h_frame, w_frame = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Strategy A: MediaPipe Tasks FaceLandmarker
        if self.use_mediapipe and self.landmarker is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = self.landmarker.detect(mp_image)

                if detection_result and detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]
                    # Landmark coords are normalized [0.0, 1.0]
                    landmark_points = np.array([[lm.x * w_frame, lm.y * h_frame] for lm in face_landmarks], dtype=np.float32)

                    # Compute bounding box from min/max coordinates with safety margin
                    x_min, y_min = np.min(landmark_points, axis=0)
                    x_max, y_max = np.max(landmark_points, axis=0)

                    w = int(x_max - x_min)
                    h = int(y_max - y_min)
                    x = int(x_min)
                    y = int(y_min)

                    margin = 0.15
                    x_pad = int(w * margin)
                    y_pad = int(h * margin)

                    x1 = max(0, x - x_pad)
                    y1 = max(0, y - y_pad)
                    x2 = min(w_frame, x + w + x_pad)
                    y2 = min(h_frame, y + h + y_pad)

                    face_crop = frame_bgr[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        face_crop = frame_bgr

                    face_crop_resized = cv2.resize(face_crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                    # Extract eye crops (64x64) for eye state classification
                    l_eye, r_eye = self._extract_eyes(frame_bgr, landmark_points, (x1, y1, x2 - x1, y2 - y1))

                    return {
                        "face_found": True,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "face_crop_bgr": face_crop_resized,
                        "left_eye_crop": l_eye,
                        "right_eye_crop": r_eye,
                        "eyes_found": (l_eye is not None and r_eye is not None),
                        "landmarks": landmark_points,
                        "detector_type": "MEDIAPIPE"
                    }
            except Exception as e:
                # Fall through to Haar cascade fallback
                pass

        # Strategy B: Haar Cascade Fallback
        if self.haar_face is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                # Pick largest face
                largest_face = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = largest_face

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_frame, x + w)
                y2 = min(h_frame, y + h)

                face_crop = frame_bgr[y1:y2, x1:x2]
                face_crop_resized = cv2.resize(face_crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                l_eye, r_eye = self._extract_eyes(frame_bgr, None, (x1, y1, x2 - x1, y2 - y1))

                return {
                    "face_found": True,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "face_crop_bgr": face_crop_resized,
                    "left_eye_crop": l_eye,
                    "right_eye_crop": r_eye,
                    "eyes_found": (l_eye is not None and r_eye is not None),
                    "landmarks": None,
                    "detector_type": "HAAR_CASCADE"
                }

        # No face detected
        return {
            "face_found": False,
            "bbox": None,
            "face_crop_bgr": None,
            "left_eye_crop": None,
            "right_eye_crop": None,
            "eyes_found": False,
            "landmarks": None,
            "detector_type": "NONE"
        }

    def _extract_eyes(self, frame_bgr, landmarks, bbox, target_size=(64, 64)):
        """Extracts left and right eye crops."""
        h_f, w_f = frame_bgr.shape[:2]
        if landmarks is not None and len(landmarks) >= 468:
            left_indices = [33, 133, 159, 145, 153, 154, 155, 144]
            right_indices = [362, 263, 386, 374, 380, 381, 382, 373]

            def crop_indices(idx_list):
                pts = landmarks[idx_list]
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                w = x_max - x_min
                h = y_max - y_min
                pad_x = int(w * 0.35)
                pad_y = int(h * 0.45)
                x1 = max(0, int(x_min - pad_x))
                y1 = max(0, int(y_min - pad_y))
                x2 = min(w_f, int(x_max + pad_x))
                y2 = min(h_f, int(y_max + pad_y))
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                return None

            l_crop = crop_indices(left_indices)
            r_crop = crop_indices(right_indices)
            return l_crop, r_crop

        # Bounding box fallback
        if bbox:
            bx, by, bw, bh = bbox
            y1 = max(0, by + int(bh * 0.20))
            y2 = min(h_f, by + int(bh * 0.50))
            mid_x = bx + bw // 2
            l_zone = frame_bgr[y1:y2, max(0, bx + int(bw * 0.08)):mid_x]
            r_zone = frame_bgr[y1:y2, mid_x:min(w_f, bx + int(bw * 0.92))]

            l_crop = cv2.resize(l_zone, target_size, interpolation=cv2.INTER_AREA) if l_zone.size > 0 else None
            r_crop = cv2.resize(r_zone, target_size, interpolation=cv2.INTER_AREA) if r_zone.size > 0 else None
            return l_crop, r_crop

        return None, None
