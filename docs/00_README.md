# Flow Matching on MNIST - Complete Pipeline

**Status:** ✓ Training Complete | Loss: 0.1829 | CPU Trained: ~5.2 hours (Ryzen 7 4700)

## Overview

This repository implements **Flow Matching (FM)** for generative modeling on MNIST, with comprehensive trajectory analysis and LDA-based visualizations.

### Key Results

- **Best Model Loss:** 0.1829 (Epoch 14)
- **Training Time:** ~5.2 hours (CPU)
- **Model Capacity:** 8.6M parameters
- **Generated Samples:** High-quality, coherent MNIST digits
- **Improvement:** 51% loss reduction (0.3647 → 0.1829)

---

## Quick Start

### 1. Download Models & Artifacts

**Create `downloads.sh`:**
```bash
#!/bin/bash

# PRIMARY: Best model + essentials
echo "Downloading main artifacts (best model)..."
gdown "GDRIVE_PRIMARY_ID" -O gdrive_main.zip
unzip -q gdrive_main.zip

# SECONDARY: All checkpoints, analysis files (optional)
echo "Downloading secondary artifacts..."
gdown "GDRIVE_SECONDARY_ID" -O gdrive_secondary.zip
unzip -q gdrive_secondary.zip -d secondary/

echo "Setup complete!"
```

**Usage:**
```bash
bash downloads.sh
```

### 2. Install & Run

```bash
pip install -r requirements.txt
python generate_samples.py          # Generate digit samples
python quick_3d_interactive.py      # Interactive 3D visualization
python lda_frame_sequence.py        # 51-frame trajectory analysis
```

---

## What We Did

### Phase 1: Initial Training
- Loss: 0.25 (3 epochs)
- Result: Poor quality, gibberish

### Phase 2: Hyperparameter Optimization
- **Learning Rate:** 0.0001 → 0.001 (10x)
- **Model Size:** 32 → 64 channels (2x)
- **Loss:** 0.3647 → 0.1829 (51% improvement)
- **Time:** ~5.2 hours on CPU

### Phase 3: Trajectory Analysis
- 11 time-stepped samples (t=0 to t=1)
- ODE convergence analysis
- Sample quality progression

### Phase 4: LDA Visualizations
- **2D:** Digit discrimination (2 components)
- **3D:** Digit (2) + Real/Fake (1 component)
- **51 Frames:** Evolution from t=0 to t=1
- **Interactive:** 3D Plotly with hover details

---

## Visualizations

### Static Images
- `generated_samples.png` - 16 MNIST digits (4×4 grid)
- `generation_process.png` - Quality vs ODE steps
- `loss_curves.png` - Loss trajectory + improvement
- `training_summary.png` - 4-panel overview
- `training_report.png` - Technical details

### Interactive
- `lda_3d_interactive_quick.html` - Open in browser
  - Rotate: Click + drag
  - Zoom: Scroll
  - Toggle: Click legend

### Trajectory Frames
- `lda_frames/frame_2d_00.png` to `frame_2d_50.png`
- `lda_frames/frame_3d_00.png` to `frame_3d_50.png`
- **Note:** Fixed LDA basis (future: add arrows + dynamic basis)

---

## Project Structure

```
├── train.py                    # Training script
├── config/train_balanced.yaml  # Balanced config (24 epochs, ~8h)
├── models/
│   ├── fm.py                  # Flow Matching model
│   └── velocity_architectures/unet.py
├── visualizations/
│   ├── generate_samples.py
│   ├── quick_3d_interactive.py
│   ├── lda_frame_sequence.py
│   └── analyze_training.py
├── results/                   # Output visualizations
│   ├── *.png                 # Static images
│   ├── *.html                # Interactive plots
│   └── lda_frames/           # 51×2 frame sequences
├── data/                      # MNIST (auto-download)
├── outputs/                   # Training logs (local)
└── README.md
```

---

## Training from Scratch

```bash
python train.py --config-name=train_balanced
```

**Configs:**
- `train_balanced` (recommended): 24 epochs, ~8 hours CPU
- `train_improved`: 150 epochs, ~50 hours CPU

---

## Performance

| Metric | Value |
|--------|-------|
| Best Loss | 0.1829 |
| Improvement | 51% |
| Time | 5.2 hours (CPU) |
| Epochs | 24/24 |
| Checkpoints | 12 (top-k) |
| Parameters | 8.6M |
| ODE Steps | 5-50 (adjustable) |

---

## Google Drive Structure

**PRIMARY FOLDER (Essential):**
- `fm-balanced-epoch=014-train_loss=0.1829.ckpt` (best model)
- Training configs and setup

**SECONDARY FOLDER (Optional):**
- All 12 checkpoints (epochs 12-23)
- TensorBoard logs
- Analysis NPZ files

---

## Reproduction

```bash
# Clone
git clone <repo>
cd flow-matching-mnist

# Download models
bash downloads.sh

# Setup
pip install -r requirements.txt

# Generate & visualize
python generate_samples.py
python quick_3d_interactive.py
python lda_frame_sequence.py

# View results
# - Static: results/*.png
# - Interactive: results/lda_3d_interactive_quick.html
# - Frames: results/lda_frames/frame_*.png
```

---

## Future Work

- [ ] **Dynamic LDA:** Refit basis per frame
- [ ] **Trajectory Arrows:** Velocity vectors
- [ ] **Animations:** Convert to MP4/GIF
- [ ] **Conditional Generation:** Class-conditional sampling

---

**Status:** ✓ Complete & Deployment Ready  
**Last Updated:** 2026-05-02
