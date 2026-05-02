#!/bin/bash
# Setup script for two-tower-recsys
# Downloads MovieLens 25M dataset and creates the Python environment

set -e

echo "=== Two-Tower RecSys Setup ==="
echo ""

# Step 1: Create virtual environment
echo "[1/3] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
else
    echo "  .venv already exists, skipping."
fi

# Step 2: Install dependencies
echo "[2/3] Installing dependencies..."
.venv/bin/pip install --quiet \
    torch \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    faiss-cpu \
    matplotlib \
    scipy \
    pyarrow \
    jupyterlab \
    ipywidgets

# Step 3: Download MovieLens 25M
echo "[3/3] Downloading MovieLens 25M dataset..."
mkdir -p data
if [ ! -d "data/ml-25m" ]; then
    echo "  Downloading from grouplens.org (~250MB compressed)..."
    curl -L -o data/ml-25m.zip "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    echo "  Extracting..."
    unzip -q data/ml-25m.zip -d data/
    rm data/ml-25m.zip
    echo "  Done. Dataset at data/ml-25m/"
else
    echo "  data/ml-25m/ already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run notebooks in order starting from 01_data_loading_and_exploration.ipynb"
echo "  2. Each notebook saves artifacts needed by subsequent notebooks"
echo "  3. Full pipeline takes ~2-3 hours on M4 Max"
echo ""
echo "  .venv/bin/jupyter lab notebooks/"
