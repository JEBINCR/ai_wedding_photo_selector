"""
Downloads the OpenCV DNN face detection model files into ./models/
These files enable the more accurate ResNet-SSD face detector.
Run once before using the pipeline.
"""

import os
import sys
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Install requests and tqdm first:  pip install requests tqdm")
    sys.exit(1)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  Already present: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  ✅ Saved to {dest}")


if __name__ == "__main__":
    print("Downloading DNN face detection model files...")
    for filename, url in FILES.items():
        download(url, MODELS_DIR / filename)
    print("\nDone! Models ready in ./models/")
