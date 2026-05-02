# Deployment Structure Guide

This document describes how to organize artifacts for Google Drive deployment.

## Overview

The project is split into two Google Drive folders:
- **PRIMARY** - Essential files needed to run the model (best checkpoint + code)
- **SECONDARY** - Additional artifacts, backups, and research files

---

## PRIMARY Folder Contents

**Purpose:** Minimal, self-contained setup for running the trained model

```
PRIMARY/
├── checkpoints/
│   └── fm-balanced-epoch=014-train_loss=0.1829.ckpt      [~103 MB - BEST MODEL]
├── config/
│   └── train_balanced.yaml
├── src/
│   ├── train.py
│   ├── models/
│   └── config/
├── visualizations/
│   ├── training/
│   │   ├── generate_samples.py
│   │   └── analyze_training.py
│   └── lda/
│       ├── quick_3d_interactive.py
│       └── lda_frame_sequence.py
├── results/
│   ├── generated_samples.png
│   ├── generation_process.png
│   ├── loss_curves.png
│   └── training_report.png
├── requirements.txt
├── README.md                            [Quick start guide]
└── INFERENCE_GUIDE.md                   [How to generate samples]
```

**Total Size:** ~300 MB
**Purpose:** Can clone + download + run inference immediately

---

## SECONDARY Folder Contents

**Purpose:** Complete training artifacts, backups, and research materials

```
SECONDARY/
├── checkpoints_all/
│   ├── fm-balanced-epoch=012-train_loss=0.1855.ckpt
│   ├── fm-balanced-epoch=013-train_loss=0.1845.ckpt
│   ├── fm-balanced-epoch=015-train_loss=0.1835.ckpt
│   ├── fm-balanced-epoch=016-train_loss=0.1831.ckpt
│   ├── fm-balanced-epoch=017-train_loss=0.1830.ckpt
│   ├── fm-balanced-epoch=018-train_loss=0.1820.ckpt
│   ├── fm-balanced-epoch=019-train_loss=0.1819.ckpt
│   ├── fm-balanced-epoch=020-train_loss=0.1799.ckpt
│   ├── fm-balanced-epoch=021-train_loss=0.1823.ckpt
│   ├── fm-balanced-epoch=022-train_loss=0.1808.ckpt
│   ├── fm-balanced-epoch=023-train_loss=0.1807.ckpt
│   └── ... [other checkpoints]
├── training_logs/
│   ├── training_improved.log
│   ├── training_progress.txt
│   └── tb_logs/
├── outputs/
│   └── fm/2026-05-01/22-58-12/training_logs/version_0/
├── results_full/
│   ├── lda_frames/
│   │   ├── frame_2d_00.png - frame_2d_50.png
│   │   └── frame_3d_00.png - frame_3d_50.png
│   └── [other visualizations]
└── RESEARCH_NOTES.md                    [Training insights and analysis]
```

**Total Size:** ~2-3 GB (checkpoints: 1.1 GB + logs + frames)
**Purpose:** Complete training record, can reproduce analysis, access alternative checkpoints

---

## Setup Instructions

### For Users (Running Inference)

1. Clone the repository
2. Run the download script for PRIMARY folder only:
   ```bash
   bash scripts/download_primary.sh
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate samples:
   ```bash
   python visualizations/training/generate_samples.py
   ```

### For Researchers (Full Analysis)

1. Clone the repository
2. Run both download scripts:
   ```bash
   bash scripts/download_primary.sh
   bash scripts/download_secondary.sh
   ```
3. Access full training logs, all checkpoints, and frame sequences
4. Analyze convergence, try alternative checkpoints, regenerate frames

---

## File Sizes Summary

| Artifact | Size | Folder |
|----------|------|--------|
| Best checkpoint | 103 MB | PRIMARY |
| Code + configs | 50 MB | PRIMARY |
| Visualizations (sample images) | 50 MB | PRIMARY |
| Other 11 checkpoints | 1.1 GB | SECONDARY |
| Training logs + TB logs | 500 MB | SECONDARY |
| LDA frame sequences (102 frames) | 1.5 GB | SECONDARY |
| **PRIMARY Total** | **~300 MB** | PRIMARY |
| **SECONDARY Total** | **~3 GB** | SECONDARY |

---

## Deployment Checklist

- [ ] Create PRIMARY folder on Google Drive
- [ ] Upload PRIMARY artifacts
- [ ] Get PRIMARY folder ID
- [ ] Create SECONDARY folder on Google Drive
- [ ] Upload SECONDARY artifacts
- [ ] Get SECONDARY folder ID
- [ ] Update scripts/download_primary.sh with PRIMARY_ID
- [ ] Update scripts/download_secondary.sh with SECONDARY_ID
- [ ] Test downloads from clean clone
- [ ] Push to GitHub
- [ ] Update README.md with drive links

---

## Notes

- PRIMARY is designed for quick setup and inference testing
- SECONDARY can be optional for most users but essential for researchers
- Both folders reference the same git repository (code is in GitHub)
- gdown will download entire folder structures automatically
