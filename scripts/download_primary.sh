#!/bin/bash

# Download PRIMARY artifacts from Google Drive
# Contains: Best checkpoint + core code + key visualizations
# Size: ~300 MB

set -e

echo "==========================================="
echo "Downloading PRIMARY artifacts (best model)"
echo "==========================================="
echo ""

# PRIMARY folder ID
PRIMARY_FOLDER_ID="1tT1cAT-PVAz9TkFhSYbFUCslKJ0x73lI"

if [ "$PRIMARY_FOLDER_ID" = "YOUR_PRIMARY_FOLDER_ID" ]; then
    echo "ERROR: PRIMARY_FOLDER_ID not set!"
    echo "Please update this script with the actual Google Drive folder ID"
    exit 1
fi

echo "Downloading primary artifacts..."
echo "Folder ID: $PRIMARY_FOLDER_ID"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

# Create checkpoint directory
mkdir -p outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints

# Download the entire PRIMARY folder
gdown --folder "$PRIMARY_FOLDER_ID" -O . --quiet

echo ""
echo "==========================================="
echo "Download complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Generate samples: python visualizations/training/generate_samples.py"
echo "3. View results in: results/"
echo ""
