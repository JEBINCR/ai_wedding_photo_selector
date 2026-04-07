"""
Face detection using OpenCV DNN (ResNet SSD) with Haar Cascade fallback.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV ships these DNN weights for face detection
DNN_PROTO = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"


class FaceDetector:
    def __init__(self, config: dict):
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.min_face_size = config.get("min_face_size", 40)
        self.net = self._load_dnn()
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> dict:
        faces = self._detect_dnn(image)
        if not faces:
            faces = self._detect_haar(image)

        return {
            "face_count": len(faces),
            "faces": faces,           # list of (x, y, w, h) tuples
        }

    # ------------------------------------------------------------------
    def _detect_dnn(self, image: np.ndarray) -> list[tuple]:
        if self.net is None:
            return []
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            fw, fh = x2 - x1, y2 - y1
            if fw >= self.min_face_size and fh >= self.min_face_size:
                faces.append((x1, y1, fw, fh))
        return faces

    def _detect_haar(self, image: np.ndarray) -> list[tuple]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []

    def _load_dnn(self) -> Optional[cv2.dnn.Net]:
        proto = Path(DNN_PROTO)
        model = Path(DNN_MODEL)
        if proto.exists() and model.exists():
            try:
                net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
                logger.info("DNN face detector loaded.")
                return net
            except Exception as e:
                logger.warning(f"DNN load failed: {e}. Using Haar fallback.")
        else:
            logger.info("DNN model files not found — using Haar Cascade for face detection.")
        return None
