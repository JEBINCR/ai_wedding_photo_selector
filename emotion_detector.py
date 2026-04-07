"""
Emotion & smile detection using DeepFace (FER / AffectNet backbone).
Falls back to a simple mouth-curvature heuristic if DeepFace is unavailable.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace loaded for emotion detection.")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not installed — using fallback smile heuristic.")


class EmotionDetector:
    POSITIVE_EMOTIONS = {"happy", "surprise"}
    NEGATIVE_EMOTIONS = {"sad", "angry", "disgust", "fear"}

    def __init__(self, config: dict):
        self.backend = config.get("backend", "opencv")
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def detect(self, image: np.ndarray, faces: list[tuple]) -> dict:
        if not faces:
            return {"dominant_emotions": {}, "avg_smile_score": 0.0}

        if DEEPFACE_AVAILABLE:
            return self._detect_deepface(image, faces)
        return self._detect_fallback(image, faces)

    # ------------------------------------------------------------------
    def _detect_deepface(self, image: np.ndarray, faces: list[tuple]) -> dict:
        emotions_per_face = []
        smile_scores = []

        for (x, y, w, h) in faces:
            face_crop = image[
                max(0, y): y + h,
                max(0, x): x + w
            ]
            if face_crop.size == 0:
                continue
            try:
                analysis = DeepFace.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]
                emotion_scores = analysis.get("emotion", {})
                dominant = analysis.get("dominant_emotion", "neutral")
                emotions_per_face.append(dominant)

                # Smile score = happy + 0.5*surprise, penalty for negative
                happy = emotion_scores.get("happy", 0) / 100.0
                surprise = emotion_scores.get("surprise", 0) / 100.0
                negative = sum(
                    emotion_scores.get(e, 0) / 100.0
                    for e in self.NEGATIVE_EMOTIONS
                )
                smile_score = min(1.0, happy + 0.3 * surprise - 0.2 * negative)
                smile_scores.append(max(0.0, smile_score))

            except Exception as e:
                logger.debug(f"DeepFace analysis failed for face crop: {e}")

        return {
            "dominant_emotions": dict(enumerate(emotions_per_face)),
            "avg_smile_score": float(np.mean(smile_scores)) if smile_scores else 0.0,
        }

    def _detect_fallback(self, image: np.ndarray, faces: list[tuple]) -> dict:
        """Use Haar smile cascade as a lightweight fallback."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smile_scores = []

        for (x, y, w, h) in faces:
            # Only look in lower half of face for smiles
            roi = gray[y + h // 2: y + h, x: x + w]
            smiles = self.smile_cascade.detectMultiScale(
                roi, scaleFactor=1.7, minNeighbors=20, minSize=(15, 15)
            )
            smile_scores.append(1.0 if len(smiles) > 0 else 0.3)

        return {
            "dominant_emotions": {},
            "avg_smile_score": float(np.mean(smile_scores)) if smile_scores else 0.0,
        }
