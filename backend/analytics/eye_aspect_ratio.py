"""
Eye Aspect Ratio (EAR) Calculation Module
Based on Soukupová and Cech (2016) landmark geometry formulation:
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
"""

import numpy as np

# Standard 6 landmark points per eye for MediaPipe Face Mesh
# Left Eye indices: [P1_outer, P2_top1, P3_top2, P4_inner, P5_bottom2, P6_bottom1]
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Right Eye indices: [P1_outer, P2_top1, P3_top2, P4_inner, P5_bottom2, P6_bottom1]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

def calculate_ear_single_eye(eye_points):
    """
    Computes EAR for 6 2D/3D points: [p1, p2, p3, p4, p5, p6]
    """
    pts = np.array(eye_points, dtype=np.float32)
    if pts.shape[0] != 6:
        raise ValueError(f"Expected 6 landmark points, got {pts.shape[0]}")

    # Vertical distances
    d_v1 = np.linalg.norm(pts[1] - pts[5])
    d_v2 = np.linalg.norm(pts[2] - pts[4])

    # Horizontal distance
    d_h = np.linalg.norm(pts[0] - pts[3])

    if d_h < 1e-6:
        return 0.0

    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return float(ear)

class EyeTracker:
    """
    Tracks eye landmarks, computes left/right and combined EAR,
    and maintains per-driver calibration.
    """
    def __init__(self, default_threshold=0.22):
        self.default_threshold = default_threshold
        self.calibrated_threshold = default_threshold
        self.is_calibrated = False
        self._calibration_samples = []

    def compute_ear(self, landmarks_dict_or_list, frame_width=1.0, frame_height=1.0):
        """
        Calculates EAR from a dictionary or list of (x, y) coordinates.
        Supports both normalized [0, 1] coordinates and pixel coordinates.
        """
        try:
            # Extract left eye points
            left_pts = [landmarks_dict_or_list[idx] for idx in LEFT_EYE_LANDMARKS]
            right_pts = [landmarks_dict_or_list[idx] for idx in RIGHT_EYE_LANDMARKS]

            left_ear = calculate_ear_single_eye(left_pts)
            right_ear = calculate_ear_single_eye(right_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            threshold = self.calibrated_threshold if self.is_calibrated else self.default_threshold
            is_closed = avg_ear < threshold

            return {
                "valid": True,
                "left_ear": round(left_ear, 3),
                "right_ear": round(right_ear, 3),
                "avg_ear": round(avg_ear, 3),
                "is_closed": is_closed,
                "threshold_used": round(threshold, 3)
            }
        except (IndexError, KeyError, TypeError):
            return {
                "valid": False,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "is_closed": False,
                "threshold_used": self.default_threshold
            }

    def start_calibration(self):
        self._calibration_samples = []
        self.is_calibrated = False

    def add_calibration_sample(self, open_eyes_ear):
        if open_eyes_ear > 0.10:
            self._calibration_samples.append(open_eyes_ear)

    def finalize_calibration(self):
        """
        Sets threshold to 75% of driver's baseline open-eye EAR.
        """
        if len(self._calibration_samples) >= 10:
            baseline = float(np.median(self._calibration_samples))
            self.calibrated_threshold = round(baseline * 0.75, 3)
            self.is_calibrated = True
            return True, self.calibrated_threshold
        return False, self.default_threshold
