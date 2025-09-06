#!/bin/bash

# GAIA Environment Setup Script
# This script helps set up the correct Python environment for GAIA

echo "🚀 GAIA Environment Setup"
echo "========================="

# Check current Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "Current Python version: $PYTHON_VERSION"

# Check if Python version is compatible with PyTorch
if [[ "$PYTHON_VERSION" > "3.12" ]]; then
    echo "⚠️  Warning: Python $PYTHON_VERSION detected"
    echo "   PyTorch currently supports Python 3.8-3.12"
    echo "   Consider installing Python 3.11 or 3.12 for better compatibility"
    echo ""
    echo "   Options:"
    echo "   1. Install Python 3.11 via Homebrew: brew install python@3.11"
    echo "   2. Use pyenv to manage Python versions: brew install pyenv"
    echo "   3. Continue with current Python (may require building from source)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch based on system
echo "🔥 Installing PyTorch..."
echo "   Detecting system configuration..."

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA GPU detected, installing CUDA version"
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "   No NVIDIA GPU detected, installing CPU version"
    python -m pip install torch torchvision torchaudio
fi

# Install other requirements
echo "📚 Installing other dependencies..."
python -m pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "   python -c 'import torch; print(f\"PyTorch {torch.__version__} installed successfully!\")'"