#!/bin/bash

# Download SECONDARY artifacts from Google Drive
# Contains: All other checkpoints, training logs, frame sequences
# Size: ~3 GB
# This script is optional - use only if you need full training artifacts

set -e

echo "==========================================="
echo "Downloading SECONDARY artifacts"
echo "==========================================="
echo "This will download:"
echo "  - All 11 other checkpoints (~1.1 GB)"
echo "  - Full training logs (~500 MB)"
echo "  - LDA frame sequences (~1.5 GB)"
echo ""
echo "Total size: ~3 GB"
echo "Time: 10-30 minutes depending on connection"
echo "==========================================="
echo ""

# SECONDARY folder ID
SECONDARY_FOLDER_ID="1Er-etthLRmheD3HsFWfCbxnHtAZSYyqz"

if [ "$SECONDARY_FOLDER_ID" = "YOUR_SECONDARY_FOLDER_ID" ]; then
    echo "ERROR: SECONDARY_FOLDER_ID not set!"
    echo "Please update this script with the actual Google Drive folder ID"
    exit 1
fi

echo "Downloading secondary artifacts..."
echo "Folder ID: $SECONDARY_FOLDER_ID"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

# Download the entire SECONDARY folder
gdown --folder "$SECONDARY_FOLDER_ID" -O . --quiet

echo ""
echo "==========================================="
echo "Download complete!"
echo "==========================================="
echo ""
echo "Now available:"
echo "  - All checkpoint files in: outputs/"
echo "  - Training logs in: training logs/"
echo "  - LDA frame sequences in: results/"
echo ""
echo "You can now:"
echo "  - Try alternative checkpoints"
echo "  - Regenerate visualizations"
echo "  - Analyze full training logs"
echo ""
