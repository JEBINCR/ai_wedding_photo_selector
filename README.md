# 💍 AI Wedding Photo Selector

Automatically analyse and select the **top 50 best wedding photos** from a folder using computer vision.

---

## ✨ Features

| Signal | Method | Library |
|---|---|---|
| Face detection | ResNet-SSD DNN + Haar Cascade fallback | OpenCV |
| Blur detection | Laplacian variance + FFT sharpness | OpenCV / NumPy |
| Smile & emotion | DeepFace (AffectNet / FER) + Haar fallback | DeepFace / TF |
| Eyes open | MediaPipe Face Mesh (EAR) + Haar fallback | MediaPipe |
| Composition/exposure | Brightness & contrast analysis | OpenCV |

---

## 📁 Project Structure

```
wedding_photo_selector/
├── main.py                    # CLI entry point
├── requirements.txt
├── download_models.py         # One-time DNN model download
├── config/
│   └── settings.yaml          # Tunable parameters
├── src/
│   ├── pipeline.py            # Orchestrates all detectors
│   ├── image_loader.py        # EXIF-aware image loading
│   ├── face_detector.py       # Face detection
│   ├── blur_detector.py       # Blur / sharpness detection
│   ├── emotion_detector.py    # Smile & emotion scoring
│   ├── eye_detector.py        # Eye openness (EAR method)
│   ├── scorer.py              # Composite score formula
│   ├── reporter.py            # HTML + CSV report generation
│   └── utils.py               # Logging helpers
├── models/                    # DNN weights (downloaded separately)
├── input/                     # ← Put your wedding photos here
├── output/
│   ├── top_photos/            # ← Selected photos copied here
│   └── reports/               # ← report.html + all_scores.csv
└── tests/
    └── test_detectors.py
```

---

## 🚀 Installation

### 1. Clone / create the project
```bash
cd wedding_photo_selector
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download DNN face-detection model (optional but recommended)
```bash
python download_models.py
```

> **Without this step** the system falls back to OpenCV Haar Cascades — still works, slightly less accurate.

---

## 🏃 Usage

### Basic — select top 50 photos
```bash
python main.py --input ./input --output ./output --top 50
```

### Custom number of selections
```bash
python main.py -i /path/to/photos -o /path/to/output -n 30
```

### All options
```
--input  / -i   Path to folder of wedding images  (default: ./input)
--output / -o   Path for results                  (default: ./output)
--top    / -n   How many photos to select         (default: 50)
--config / -c   Path to settings.yaml             (default: ./config/settings.yaml)
--log-level     DEBUG | INFO | WARNING | ERROR    (default: INFO)
```

---

## 📊 Scoring Formula

Each photo is scored 0–100 using a weighted composite:

| Criterion | Weight | How it's measured |
|---|---|---|
| **Sharpness** | 30% | Laplacian variance + FFT HF energy |
| **Face presence** | 25% | Face count relative to ideal (2) |
| **Eyes open** | 20% | Eye Aspect Ratio via MediaPipe landmarks |
| **Smile / emotion** | 15% | DeepFace happiness/surprise score |
| **Composition** | 10% | Brightness centering + tonal contrast |

Weights can be adjusted in `src/scorer.py`.

---

## 🔧 Configuration (`config/settings.yaml`)

```yaml
blur_detector:
  blur_threshold: 80.0    # Lower = more lenient about blur

scorer:
  ideal_face_count: 2     # 2 = wedding couple; increase for group shots
  blur_good: 120          # Laplacian score above this = full sharpness points

eye_detector:
  ear_threshold: 0.20     # Lower = more lenient about closed eyes
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📦 Required Libraries

```
opencv-python    — face, eye, blur, smile detection
Pillow           — image loading with EXIF rotation
numpy            — numerical computations
deepface         — deep emotion/smile detection
tensorflow       — DeepFace backend
mediapipe        — accurate eye-open detection (Face Mesh)
PyYAML           — configuration file parsing
requests, tqdm   — DNN model download
```

---

## 🖼️ Supported Image Formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp`

---

## 📋 Output

After running you will find:

- **`output/top_photos/`** — top N images copied and ranked (`rank_001_filename.jpg`)
- **`output/reports/report.html`** — visual HTML report with scores, badges, and highlights
- **`output/reports/all_scores.csv`** — full CSV for all analysed images
