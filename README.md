# SmartApp-Usage-Extractor -- YOLO-to-OCR Smartphone App Usage Pipeline
Undergraduate research project conducted for Dr. Dougall at UTA, combining computer vision and OCR to analyze smartphone app usage patterns. Developed as a capstone project, this pipeline detects app interfaces using custom YOLO models and extracts usage data through multi-stage OCR processing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
