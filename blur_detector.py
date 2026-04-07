"""
Blur detection using Laplacian variance and FFT magnitude analysis.
Higher variance = sharper image.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BlurDetector:
    def __init__(self, config: dict):
        self.blur_threshold = config.get("blur_threshold", 80.0)
        self.use_fft = config.get("use_fft", True)

    def detect(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Primary: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Secondary: FFT high-frequency energy
        fft_score = self._fft_sharpness(gray) if self.use_fft else 0.0

        # Combine: 70% laplacian, 30% FFT
        combined = laplacian_var * 0.7 + fft_score * 0.3
        is_blurry = combined < self.blur_threshold

        return {
            "laplacian_variance": float(laplacian_var),
            "fft_score": float(fft_score),
            "combined_sharpness": float(combined),
            "is_blurry": is_blurry,
        }

    @staticmethod
    def _fft_sharpness(gray: np.ndarray) -> float:
        """High-frequency energy via FFT — blurry images have less HF content."""
        h, w = gray.shape
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        # Mask the center (low freq) and measure the rest
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 8
        mask = np.ones((h, w), dtype=bool)
        Y, X = np.ogrid[:h, :w]
        mask[(Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2] = False
        hf_energy = magnitude[mask].mean()
        return float(hf_energy)
