#!/bin/bash
# Install dependencies for the Offline LaTeX PDF Conversion Pipeline
echo "Installing dependencies..."
pip install "docling[vlm,tesserocr]>=1.0.0"
pip install "nougat-ocr>=0.1.17"
pip install "pix2tex>=0.1.0"
echo "Done. Please ensure you have Tesseract and Poppler installed on your system (e.g., dnf install tesseract tesseract-devel poppler-cpp)."
