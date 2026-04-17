#!/bin/bash

# NLP spaCy Environment Setup Script
# Creates a fresh Python 3.10 virtual environment with all required dependencies

set -e  # Exit on any error

echo "=== NLP spaCy Environment Setup ==="
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv ex00

# Activate virtual environment
echo "Activating virtual environment..."
source ex00/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install setuptools and wheel (required for proper package installation)
echo "Installing setuptools and wheel..."
pip install setuptools wheel

# Install core dependencies
echo "Installing core dependencies..."
pip install spacy==3.4.4 numpy==1.24.3 pandas jupyter scikit-learn matplotlib

# Download spaCy language models
echo "Downloading en_core_web_sm model..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1.tar.gz

echo "Downloading en_core_web_md model..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.1/en_core_web_md-3.4.1.tar.gz

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import spacy; print(f'spaCy version: {spacy.__version__}')"
python -c "nlp = spacy.load('en_core_web_sm'); print('✓ en_core_web_sm loaded')"
python -c "nlp = spacy.load('en_core_web_md'); print('✓ en_core_web_md loaded')"
python -c "import pandas; print(f'✓ Pandas version: {pandas.__version__}')"
python -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source ex00/bin/activate"
echo "Run Jupyter with: jupyter notebook"
