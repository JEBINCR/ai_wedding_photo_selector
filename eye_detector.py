"""
Eye openness detection using:
  1. MediaPipe Face Mesh — Eye Aspect Ratio (EAR) method (preferred)
  2. Haar eye cascade fallback
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe loaded for eye detection.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed — using Haar eye cascade fallback.")

# MediaPipe Face Mesh landmark indices for left/right eye
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20   # below this → eye considered closed


def eye_aspect_ratio(landmarks, eye_indices, img_w: int, img_h: int) -> float:
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in eye_indices
    ])
    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


class EyeDetector:
    def __init__(self, config: dict):
        self.ear_threshold = config.get("ear_threshold", EAR_THRESHOLD)
        if MEDIAPIPE_AVAILABLE:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, faces: list[tuple]) -> dict:
        if MEDIAPIPE_AVAILABLE:
            return self._detect_mediapipe(image)
        return self._detect_haar(image, faces)

    # ------------------------------------------------------------------
    def _detect_mediapipe(self, image: np.ndarray) -> dict:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        results = self._mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"all_eyes_open": False, "open_ratio": 0.0, "ear_values": []}

        ear_values = []
        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ear_values.append((left_ear + right_ear) / 2.0)

        open_count = sum(1 for e in ear_values if e >= self.ear_threshold)
        open_ratio = open_count / len(ear_values) if ear_values else 0.0

        return {
            "all_eyes_open": open_ratio >= 0.8,
            "open_ratio": float(open_ratio),
            "ear_values": [float(e) for e in ear_values],
        }

    def _detect_haar(self, image: np.ndarray, faces: list[tuple]) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        open_count, total = 0, 0

        for (x, y, w, h) in faces:
            # Look only in top 60% of face region for eyes
            roi = gray[y: y + int(h * 0.6), x: x + w]
            eyes = self._eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            total += 2  # expect 2 eyes per face
            open_count += min(len(eyes), 2)

        open_ratio = open_count / total if total else 0.0
        return {
            "all_eyes_open": open_ratio >= 0.8,
            "open_ratio": float(open_ratio),
            "ear_values": [],
        }
