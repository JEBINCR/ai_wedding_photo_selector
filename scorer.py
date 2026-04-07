"""
Composite photo scorer.

Score breakdown (0-100):
  - Sharpness (blur)        : 30 pts
  - Face presence & count   : 25 pts
  - Eyes open               : 20 pts
  - Smile / emotion         : 15 pts
  - Composition & exposure  : 10 pts
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PhotoScorer:
    # Weights (must sum to 1.0)
    W_SHARPNESS   = 0.30
    W_FACE        = 0.25
    W_EYES        = 0.20
    W_SMILE       = 0.15
    W_COMPOSITION = 0.10

    def __init__(self, config: dict):
        self.ideal_face_count   = config.get("ideal_face_count", 2)
        self.blur_min           = config.get("blur_min", 0)
        self.blur_max           = config.get("blur_max", 500)
        self.blur_good          = config.get("blur_good", 120)   # score=1 above this

    # ------------------------------------------------------------------
    def score(self, result: dict) -> float:
        s_sharp = self._sharpness_score(result)
        s_face  = self._face_score(result)
        s_eyes  = self._eye_score(result)
        s_smile = result.get("smile_score", 0.0)
        s_comp  = self._composition_score(result)

        total = (
            s_sharp   * self.W_SHARPNESS +
            s_face    * self.W_FACE      +
            s_eyes    * self.W_EYES      +
            s_smile   * self.W_SMILE     +
            s_comp    * self.W_COMPOSITION
        )
        return round(float(total) * 100, 2)

    # ------------------------------------------------------------------
    def _sharpness_score(self, result: dict) -> float:
        lv = result.get("blur_score", 0.0)
        if lv >= self.blur_good:
            return 1.0
        return float(np.clip((lv - self.blur_min) / (self.blur_good - self.blur_min + 1e-6), 0, 1))

    def _face_score(self, result: dict) -> float:
        n = result.get("face_count", 0)
        if n == 0:
            return 0.05   # rare: great landscape shot without faces still gets tiny credit
        if n <= self.ideal_face_count:
            return 1.0
        # Diminishing returns for very crowded group shots
        return max(0.3, 1.0 - (n - self.ideal_face_count) * 0.05)

    def _eye_score(self, result: dict) -> float:
        if result.get("face_count", 0) == 0:
            return 0.5   # no faces → neutral
        return result.get("eye_open_ratio", 0.0)

    def _composition_score(self, result: dict) -> float:
        """
        Quick proxy metrics:
        - Brightness: penalise over/under-exposed images
        - Contrast: reward well-separated tones
        """
        path = result.get("path")
        if path is None:
            return 0.5
        try:
            img = cv2.imread(str(path))
            if img is None:
                return 0.5
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean() / 255.0
            # Ideal brightness ~0.45-0.65
            brightness_score = 1.0 - abs(mean_brightness - 0.55) * 2.5
            brightness_score = float(np.clip(brightness_score, 0, 1))

            std_contrast = gray.std() / 128.0
            contrast_score = float(np.clip(std_contrast, 0, 1))

            return 0.6 * brightness_score + 0.4 * contrast_score
        except Exception:
            return 0.5
