#!/bin/bash

# Flow Matching MNIST - Download Script
# Downloads models and artifacts from Google Drive

set -e

echo "=========================================="
echo "Flow Matching MNIST - Setup"
echo "=========================================="
echo

# Primary: Best model + essentials
PRIMARY_ID="YOUR_PRIMARY_GDRIVE_ID"
SECONDARY_ID="YOUR_SECONDARY_GDRIVE_ID"

if [ "$PRIMARY_ID" = "YOUR_PRIMARY_GDRIVE_ID" ]; then
    echo "ERROR: Please set PRIMARY_ID and SECONDARY_ID in downloads.sh"
    echo
    echo "To get Google Drive folder IDs:"
    echo "1. Open folder in Google Drive"
    echo "2. Copy ID from URL: https://drive.google.com/drive/folders/FOLDER_ID"
    echo "3. Replace PRIMARY_ID and SECONDARY_ID in this script"
    exit 1
fi

echo "[1/3] Downloading primary artifacts (best model, essentials)..."
gdown "$PRIMARY_ID" -O gdrive_main.zip --quiet
unzip -q gdrive_main.zip -d ./
echo "[OK] Primary artifacts downloaded"
echo

echo "[2/3] Downloading secondary artifacts (checkpoints, logs)..."
gdown "$SECONDARY_ID" -O gdrive_secondary.zip --quiet
unzip -q gdrive_secondary.zip -d ./secondary/
echo "[OK] Secondary artifacts downloaded"
echo

echo "[3/3] Verifying setup..."
if [ -f "outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt" ]; then
    echo "[OK] Best model checkpoint found"
else
    echo "[WARN] Best model checkpoint not found"
fi

if [ -d "config" ]; then
    echo "[OK] Config directory found"
else
    echo "[WARN] Config directory not found"
fi

echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. pip install -r requirements.txt"
echo "2. python generate_samples.py"
echo "3. python quick_3d_interactive.py"
echo "4. python lda_frame_sequence.py"
echo

