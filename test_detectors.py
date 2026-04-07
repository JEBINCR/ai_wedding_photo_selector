"""
Unit tests for individual detector modules.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import cv2


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def make_blank_image(h=300, w=300, color=(200, 200, 200)) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def make_sharp_image(h=300, w=300) -> np.ndarray:
    img = make_blank_image(h, w)
    # Draw high-contrast edges
    for i in range(0, w, 10):
        cv2.line(img, (i, 0), (i, h), (0, 0, 0), 1)
    return img


def make_blurry_image(h=300, w=300) -> np.ndarray:
    img = make_sharp_image(h, w)
    return cv2.GaussianBlur(img, (51, 51), 0)


# ------------------------------------------------------------------
# Blur detector tests
# ------------------------------------------------------------------
class TestBlurDetector:
    def setup_method(self):
        from src.blur_detector import BlurDetector
        self.detector = BlurDetector({"blur_threshold": 80.0, "use_fft": True})

    def test_sharp_image_not_blurry(self):
        img = make_sharp_image()
        result = self.detector.detect(img)
        assert result["is_blurry"] is False
        assert result["laplacian_variance"] > 80

    def test_blurry_image_is_blurry(self):
        img = make_blurry_image()
        result = self.detector.detect(img)
        assert result["is_blurry"] is True
        assert result["laplacian_variance"] < 80

    def test_returns_expected_keys(self):
        img = make_blank_image()
        result = self.detector.detect(img)
        for key in ("laplacian_variance", "fft_score", "combined_sharpness", "is_blurry"):
            assert key in result


# ------------------------------------------------------------------
# Scorer tests
# ------------------------------------------------------------------
class TestPhotoScorer:
    def setup_method(self):
        from src.scorer import PhotoScorer
        self.scorer = PhotoScorer({})

    def _make_result(self, **overrides):
        base = {
            "path": None,
            "face_count": 2,
            "blur_score": 200.0,
            "is_blurry": False,
            "smile_score": 0.9,
            "eyes_open": True,
            "eye_open_ratio": 1.0,
        }
        base.update(overrides)
        return base

    def test_perfect_image_high_score(self):
        result = self._make_result()
        score = self.scorer.score(result)
        assert score >= 70, f"Expected >=70, got {score}"

    def test_blurry_no_faces_low_score(self):
        result = self._make_result(face_count=0, blur_score=10.0, is_blurry=True,
                                   smile_score=0.0, eyes_open=False, eye_open_ratio=0.0)
        score = self.scorer.score(result)
        assert score < 40, f"Expected <40, got {score}"

    def test_score_range(self):
        result = self._make_result()
        score = self.scorer.score(result)
        assert 0 <= score <= 100


# ------------------------------------------------------------------
# Face detector (basic smoke test — no real faces in test images)
# ------------------------------------------------------------------
class TestFaceDetector:
    def setup_method(self):
        from src.face_detector import FaceDetector
        self.detector = FaceDetector({})

    def test_blank_image_no_faces(self):
        img = make_blank_image()
        result = self.detector.detect(img)
        assert result["face_count"] == 0
        assert isinstance(result["faces"], list)


# ------------------------------------------------------------------
# Image loader
# ------------------------------------------------------------------
class TestImageLoader:
    def setup_method(self):
        from src.image_loader import ImageLoader
        self.loader = ImageLoader()

    def test_missing_file_returns_none(self, tmp_path):
        result = self.loader.load(tmp_path / "nonexistent.jpg")
        assert result is None

    def test_valid_image_loads(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = make_sharp_image()
        cv2.imwrite(str(img_path), img)
        loaded = self.loader.load(img_path)
        assert loaded is not None
        assert loaded.shape == img.shape
