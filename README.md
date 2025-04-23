# SmartApp-Usage-Extractor

[Build in Progress!]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

YOLO-to-OCR Smartphone App Usage Pipeline

An end-to-end computer vision pipeline that detects smartphone app interfaces using YOLO object detection and extracts usage statistics through OCR text recognition.

![Pipeline Visualization](docs/pipeline_diagram.png)

## 🚀 Key Features

- **YOLO v11 Detection**: Custom-trained model identifies:
  - App icons
  - App name text regions
  - Usage time displays
  - Device ID markers
  
- **OCR Processing**:
  - PaddleOCR text extraction
  - Multi-stage text correction
  - Fuzzy name matching
  - Usage time validation

- **Smart Layout Analysis**:
  - Grid pattern recognition
  - Spatial relationship mapping
  - Reference line detection

- **Multi-Format Outputs**:
  - CSV reports
  - SQL database storage
  - Organized folder structure
  - Debug visualizations (toggleable)

## 📂 Output Structure
results/

├── id_folders/

│   └── ID_OWL1234/

│       ├── original_image.jpg

│       └── ocr_results.txt

├── debug_visualizations/  # Only when debug=true

│   └── debug_image.jpg

├── usage_report.csv

└── usage.db               # SQLite database


CSV Output Example

id,total_usage,Facebook,Instagram,WhatsApp

OWL1234,2h30m,45m,1h15m,30m

## Model Information
# Custom YOLO Model

- Trained on 3,376 smartphone screenshots
- Classes: app_icon, app_name, app_usage, id
- mAP@0.5: 0.99
- Input resolution: 1040x1040
  
# Model Directory Structure
models/

└── yolo/

    ├── args.yaml
    
    └── weights/
    
        └── best.pt      # Trained weights (Git LFS)

## Troubleshooting
# Current version
- Needs to improve OCR accuracy

## 📦 Installation

```bash
# Clone with Git LFS for model weights
git lfs install
git clone https://github.com/yourusername/yolo-ocr-pipeline
cd yolo-ocr-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration for PaddleOCR
pip install paddlepaddle-gpu

## 🛠️ Usage
# Process single image
python scripts/process_image.py \
  --input sample.jpg \
  --output results/ \
  --config configs/default.yaml

# Batch processing
python scripts/batch_process.py \
  --input-dir screenshots/ \
  --output-dir results/ \
  --batch-size 64
