"""
Pipeline orchestrator — wires all detectors together and scores images.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .image_loader import ImageLoader
from .face_detector import FaceDetector
from .blur_detector import BlurDetector
from .emotion_detector import EmotionDetector
from .eye_detector import EyeDetector
from .scorer import PhotoScorer
from .reporter import ReportGenerator

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class WeddingPhotoSelectorPipeline:
    def __init__(self, config_path: str, output_dir: Path):
        self.output_dir = output_dir
        self.config = self._load_config(config_path)

        logger.info("Initialising detectors...")
        self.loader = ImageLoader()
        self.face_detector = FaceDetector(self.config.get("face_detector", {}))
        self.blur_detector = BlurDetector(self.config.get("blur_detector", {}))
        self.emotion_detector = EmotionDetector(self.config.get("emotion_detector", {}))
        self.eye_detector = EyeDetector(self.config.get("eye_detector", {}))
        self.scorer = PhotoScorer(self.config.get("scorer", {}))
        self.reporter = ReportGenerator()
        logger.info("All detectors ready.")

    # ------------------------------------------------------------------
    def run(
        self,
        input_dir: Path,
        top_n: int = 50,
        copy_photos: bool = True,
        generate_report: bool = True,
    ) -> dict:
        image_paths = self._collect_images(input_dir)
        logger.info(f"Found {len(image_paths)} images in {input_dir}")

        if not image_paths:
            logger.warning("No images found. Check --input path and file extensions.")
            return {"total_processed": 0, "selected_photos": []}

        results = []
        for idx, img_path in enumerate(image_paths, 1):
            logger.info(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
            result = self._process_single(img_path)
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        selected = results[:top_n]

        if copy_photos:
            self._copy_selected(selected)

        if generate_report:
            self.reporter.generate(
                all_results=results,
                selected=selected,
                output_dir=self.output_dir / "reports",
            )

        return {
            "total_processed": len(results),
            "selected_photos": selected,
            "all_results": results,
        }

    # ------------------------------------------------------------------
    def _process_single(self, img_path: Path) -> dict:
        result = {
            "path": img_path,
            "filename": img_path.name,
            "score": 0.0,
            "face_count": 0,
            "blur_score": 0.0,
            "is_blurry": True,
            "emotions": {},
            "smile_score": 0.0,
            "eyes_open": False,
            "eye_open_ratio": 0.0,
            "error": None,
        }

        try:
            image = self.loader.load(img_path)
            if image is None:
                result["error"] = "Could not load image"
                return result

            # Face detection
            face_data = self.face_detector.detect(image)
            result["face_count"] = face_data["face_count"]
            result["faces"] = face_data["faces"]

            # Blur detection (whole image)
            blur_data = self.blur_detector.detect(image)
            result["blur_score"] = blur_data["laplacian_variance"]
            result["is_blurry"] = blur_data["is_blurry"]

            if face_data["face_count"] > 0:
                # Emotion + smile (per face, averaged)
                emotion_data = self.emotion_detector.detect(image, face_data["faces"])
                result["emotions"] = emotion_data["dominant_emotions"]
                result["smile_score"] = emotion_data["avg_smile_score"]

                # Eye detection
                eye_data = self.eye_detector.detect(image, face_data["faces"])
                result["eyes_open"] = eye_data["all_eyes_open"]
                result["eye_open_ratio"] = eye_data["open_ratio"]

            # Final composite score
            result["score"] = self.scorer.score(result)

        except Exception as exc:
            logger.error(f"Error processing {img_path.name}: {exc}", exc_info=True)
            result["error"] = str(exc)

        logger.debug(
            f"  faces={result['face_count']}  blur={result['blur_score']:.1f}"
            f"  smile={result['smile_score']:.2f}  eyes_open={result['eyes_open']}"
            f"  SCORE={result['score']:.2f}"
        )
        return result

    # ------------------------------------------------------------------
    def _collect_images(self, folder: Path) -> list[Path]:
        paths = []
        for ext in SUPPORTED_EXTENSIONS:
            paths.extend(folder.glob(f"*{ext}"))
            paths.extend(folder.glob(f"*{ext.upper()}"))
        return sorted(set(paths))

    def _copy_selected(self, selected: list[dict]):
        dest = self.output_dir / "top_photos"
        dest.mkdir(exist_ok=True)
        for rank, item in enumerate(selected, 1):
            src = item["path"]
            dst = dest / f"rank_{rank:03d}_{src.name}"
            shutil.copy2(src, dst)
        logger.info(f"Copied {len(selected)} photos to {dest}")

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found at {path}, using defaults.")
            return {}
