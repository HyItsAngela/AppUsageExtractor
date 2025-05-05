# AppUsageExtractor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![Project Status](https://img.shields.io/badge/status-active%20development-yellow)

> [!NOTE]
> This project is currently under active development. Core functionality is implemented with ongoing improvements.

An end-to-end computer vision pipeline that detects smartphone app interfaces using YOLO object detection and extracts usage statistics through OCR text recognition.

![Pipeline Visualization](docs/pipeline_diagram.png)

## Table of Contents
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Output Details](#-output-structure)
- [Model Information](#-model-information)
- [Installation](#-installation)
- [Usage](#-usage)

## 💡 Key Features

### Detection & Recognition
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

### Layout Understanding
- **Smart Layout Analysis**:
  - Grid pattern recognition
  - Spatial relationship mapping
  - Reference line detection

### Output System
- **Multi-Format Outputs**:
  - CSV reports
  - SQL database storage
  - Organized folder structure
  - Debug visualizations (toggleable)

## 📂 Project Structure
```text
SmartApp-Usage-Extractor/
├── configs/              # Configuration templates
├── data/                 # Sample inputs
├── docs/                 # Documentation 
├── models/
│   └── yolo/            # Custom detection model
│       ├── args.yaml    # Training configuration
│       └── weights/     # Model parameters (Git LFS)
├── results/             # Processing outputs
├── scripts/             # Execution scripts
└── src/                 # Core pipeline code
```

## 💾 Output Structure
```text
results/
├── id_folders/
│   └── ID_OWL1234/             # Example folder for a specific ID
│       ├── original_image.jpg  # Original input image
│       └── ocr_results.txt     # Extracted text results
├── debug_visualizations/     # Optional: Contains images with bounding boxes if debug=true
│   └── debug_image_ID_OWL1234.jpg # Example debug output
├── usage_report.csv          # Aggregated usage data in CSV format
└── usage.db                  # Aggregated usage data in SQLite database
```


### CSV Report Example
```text
| device_id | total_usage |   Facebook   |   Snapchat   |    WhatsApp   |    Calendar   |
|-----------|-------------|--------------|--------------|---------------|---------------|
| OWL1234   | 4h 15m      | 2h 30m       | 1h 15m       | 30m           | 0m            |
| OWL5678   | 6h 22m      | 3h 45m       | 0h 45m       | 1h 12m        | 40m           |
| OWL9012   | 3h 08m      | 1h 15m       | 1h 30m       | 15m           | 8m            |
```

## 📊 Model Information
**Custom YOLO Model**

- Trained on 3,376 smartphone screenshots
- Classes: app_icon, app_name, app_usage, id
- mAP@0.5: 0.99
- Input resolution: 1040x1040
  
### Model Directory Structure
```text
models/
└── yolo/
    ├── args.yaml     # Training configuration
    └── weights/
        └── best.pt   # Trained model weights (Tracked with Git LFS)
```

> [!IMPORTANT]
> The best.pt model weights file is managed using Git Large File Storage (LFS). Ensure you have Git LFS installed (git lfs install) before cloning to download the model file correctly.

> [!CAUTION] 
> ⚠ Current version Limitations:
> Needs to improve OCR accuracy for low quality images.

## 📦 Installation
```bash
# 1. Clone with Git LFS for model weights
git lfs install
git clone https://github.com/HyItsAngela/SmartApp-Usage-Extractor
cd SmartApp-Usage-Extractor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: GPU acceleration for PaddleOCR
pip install paddlepaddle-gpu
```

## 🛠 Usage
```bash
# Process Single Image:
python scripts/process_image.py \
  --input sample.jpg \
  --output results/ \
  --config configs/default.yaml

# Process a Directory of Images (Batch Processing):
python scripts/batch_process.py \
  --input-dir path/to/screenshots/ \
  --output-dir results/ \
  --config configs/default.yaml \
  --batch-size 16 \

