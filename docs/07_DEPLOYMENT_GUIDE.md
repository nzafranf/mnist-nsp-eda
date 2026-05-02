# Deployment Guide: GitHub + Google Drive

## Overview

This project is organized for easy distribution:
- **GitHub:** All code + visualization examples
- **Google Drive (PRIMARY):** Best model + essentials
- **Google Drive (SECONDARY):** Checkpoints + analysis files

---

## GitHub Repository Structure

```
flow-matching-mnist/
├── train.py                          # Training entry point
├── config/                           # Training configs
│   └── train_balanced.yaml
├── models/                           # Model definitions
│   ├── fm.py
│   └── velocity_architectures/unet.py
├── visualizations/                   # Visualization scripts
│   ├── generate_samples.py
│   ├── quick_3d_interactive.py
│   ├── lda_frame_sequence.py
│   └── analyze_training.py
├── results/                          # Example outputs (committed to git)
│   ├── generated_samples.png
│   ├── generation_process.png
│   ├── loss_curves.png
│   ├── training_summary.png
│   ├── training_report.png
│   ├── trajectory_evolution.png
│   └── lda_3d_interactive_quick.html
├── requirements.txt                  # Dependencies
├── README.md                         # Main documentation
├── downloads.sh                      # Download script
├── .gitignore
└── DEPLOYMENT_GUIDE.md
```

---

## Google Drive: PRIMARY FOLDER

**Size:** ~200-300 MB  
**Contains:** Best checkpoint + essential configs

```
PRIMARY_FOLDER/
├── outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/
│   └── fm-balanced-epoch=014-train_loss=0.1829.ckpt [98 MB]
└── config/
    ├── train_balanced.yaml
    ├── train_improved.yaml
    └── model/flow_matching_improved.yaml
```

---

## Google Drive: SECONDARY FOLDER

**Size:** ~1-2 GB  
**Contains:** All checkpoints + frame sequences + analysis files

```
SECONDARY_FOLDER/
├── outputs/fm/2026-05-01/22-58-12/training_logs/version_0/
│   ├── checkpoints/ [all 12 checkpoints]
│   └── events.out.tfevents.*
├── results/lda_frames/ [102 frame images]
└── analysis_artifacts/
```

---

## Usage

### Clone & Setup

```bash
git clone <github-repo>
cd flow-matching-mnist

# Download models from Google Drive
bash downloads.sh

# Install dependencies
pip install -r requirements.txt

# Run visualizations
python generate_samples.py
python quick_3d_interactive.py
```

### To Set Google Drive IDs

1. Upload folders to Google Drive
2. Copy folder IDs from URL: `https://drive.google.com/drive/folders/FOLDER_ID`
3. Edit downloads.sh:
   ```bash
   PRIMARY_ID="YOUR_ID"
   SECONDARY_ID="YOUR_ID"
   ```

---

## File Checklist for GitHub

- ✓ train.py
- ✓ models/ (fm.py, unet.py)
- ✓ config/ (*.yaml files)
- ✓ visualizations/ (all scripts)
- ✓ results/ (example outputs)
- ✓ requirements.txt
- ✓ README.md
- ✓ downloads.sh
- ✓ .gitignore

## Exclude from GitHub (in .gitignore)

- ✗ *.ckpt (checkpoints)
- ✗ outputs/ (training logs)
- ✗ data/ (MNIST)
- ✗ results/lda_frames/ (frame sequences)
- ✗ *.log

---

**Status:** Ready for GitHub + Google Drive deployment  
**Last Updated:** 2026-05-02
