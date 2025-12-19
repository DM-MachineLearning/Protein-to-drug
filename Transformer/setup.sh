#!/bin/bash
# Installation and setup script for Protein-to-Drug generation pipeline

set -e

echo "=========================================="
echo "Protein-to-Drug Generation Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo ""
echo "Installing PyTorch..."
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Optional: Install protein language models
echo ""
echo "Installing optional protein language models..."
pip install fair-esm  # For ESM2

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p results
mkdir -p checkpoints

# Download sample data check
echo ""
echo "Checking for data files..."
if [ -d "../CPI/CPI" ]; then
    echo "✓ CPI data directory found"
    echo "  - $(wc -l < ../CPI/CPI/smiles.smi) SMILES compounds"
    echo "  - $(wc -l < ../CPI/CPI/uniprot_ID.smi) protein IDs"
else
    echo "⚠ CPI data directory not found at ../CPI/CPI"
    echo "  Please ensure your data is in the correct location"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  python main.py --stage preprocess"
echo "  python main.py --stage train --epochs 50"
echo "  python main.py --stage generate"
echo ""
echo "Or run the full pipeline:"
echo "  python main.py --stage all"
echo ""
