#!/bin/bash
# Quick setup script for local Docling deployment

set -e

echo "======================================================================"
echo "Docling Local Setup"
echo "======================================================================"

# Check Python
echo ""
echo "[1/6] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Tesseract
echo ""
echo "[2/6] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version | head -n1)
    echo "✓ $TESSERACT_VERSION"
else
    echo "⚠ Tesseract not found"
    echo "Install: sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng (Linux)"
    echo "         brew install tesseract tesseract-lang (macOS)"
    read -p "Continue without Tesseract? (OCR will not work) [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "[3/6] Installing Python dependencies..."
cd services/ingestion
pip install -r requirements.txt
cd ../..
echo "✓ Dependencies installed"

# Download models
echo ""
echo "[4/6] Downloading Docling models..."
echo "This will download ~500MB of models (one-time only)"
read -p "Download models now? [Y/n]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    python services/ingestion/download_models.py
    echo "✓ Models downloaded"
else
    echo "⚠ Skipped. Models will download on first use."
fi

# Create directories
echo ""
echo "[5/6] Creating directories..."
mkdir -p data/models/docling
mkdir -p data/uploads
echo "✓ Directories created"

# Test offline mode
echo ""
echo "[6/6] Testing offline mode..."
if [ -f "test_offline_mode.py" ]; then
    python test_offline_mode.py
else
    echo "⚠ Test script not found, skipping..."
fi

# Summary
echo ""
echo "======================================================================"
echo "✓ Setup complete!"
echo "======================================================================"
echo ""
echo "To run locally:"
echo "  cd services/ingestion"
echo "  uvicorn api:app --host 0.0.0.0 --port 8001"
echo ""
echo "To test:"
echo "  curl -F 'file=@tests/pdf/Knoleges.pdf' http://localhost:8001/ingest"
echo ""
echo "For Docker:"
echo "  make -f Makefile.docling setup-docker"
echo ""
echo "Models location: ~/.cache/huggingface/hub/"
echo "Cache size: $(du -sh ~/.cache/huggingface/hub/ 2>/dev/null | cut -f1 || echo 'N/A')"
echo "======================================================================"

