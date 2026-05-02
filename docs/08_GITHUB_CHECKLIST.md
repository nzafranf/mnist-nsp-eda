# GitHub Push Checklist

## Files to Commit & Push to GitHub

### Core Code Files ✓
- [x] `train.py` - Training script
- [x] `models/fm.py` - Flow Matching model
- [x] `models/velocity_architectures/unet.py` - UNet architecture
- [x] `config/train_balanced.yaml` - Training config
- [x] `config/train_improved.yaml` - Alternative config
- [x] `config/model/flow_matching_improved.yaml` - Model config

### Visualization Scripts ✓
- [x] `visualizations/generate_samples.py` - Sample generation
- [x] `visualizations/quick_3d_interactive.py` - Interactive 3D
- [x] `visualizations/lda_frame_sequence.py` - Frame sequences
- [x] `visualizations/analyze_training.py` - Loss analysis
- [x] `visualizations/lda_trajectory_visualization.py` - Full animations (WIP)

### Documentation ✓
- [x] `README.md` - Main documentation
- [x] `DEPLOYMENT_GUIDE.md` - Deployment instructions
- [x] `GITHUB_PUSH_CHECKLIST.md` - This file

### Configuration Files ✓
- [x] `requirements.txt` - Dependencies
- [x] `.gitignore` - Ignore rules
- [x] `downloads.sh` - Google Drive download script

### Example Outputs (Keep Small) ✓
- [x] `results/generated_samples.png` (45 KB)
- [x] `results/generation_process.png` (39 KB)
- [x] `results/loss_curves.png` (63 KB)
- [x] `results/training_summary.png` (117 KB)
- [x] `results/training_report.png` (359 KB)
- [x] `results/trajectory_evolution.png` (27 KB)
- [x] `results/lda_3d_interactive_quick.html` (4.7 MB)

### Do NOT Commit (in .gitignore)
- [ ] *.ckpt (too large)
- [ ] outputs/ (training logs)
- [ ] data/ (MNIST dataset)
- [ ] results/lda_frames/ (102 frame images)
- [ ] *.log (training logs)
- [ ] *.npz (analysis files)

---

## Pre-Push Steps

1. **Verify .gitignore is correct**
   ```bash
   git status
   # Should NOT show: *.ckpt, outputs/, data/, results/lda_frames/
   ```

2. **Add all code files**
   ```bash
   git add train.py
   git add models/
   git add config/
   git add visualizations/
   git add results/*.png results/*.html
   git add README.md DEPLOYMENT_GUIDE.md
   git add requirements.txt .gitignore downloads.sh
   ```

3. **Verify what's staged**
   ```bash
   git status
   # Should show only code, configs, and small example outputs
   ```

4. **Commit**
   ```bash
   git commit -m "Initial commit: Flow Matching MNIST with LDA visualizations

   - Trained model (loss=0.1829) on MNIST using Flow Matching
   - Training scripts with configurable hyperparameters
   - Visualization pipeline: samples, trajectories, LDA-based analysis
   - 51-frame trajectory sequence (2D + 3D)
   - Interactive 3D Plotly visualization
   - Complete documentation and deployment guide
   - Models and checkpoints on Google Drive (via downloads.sh)"
   ```

5. **Push to GitHub**
   ```bash
   git push origin main
   ```

---

## Google Drive Preparation

### PRIMARY Folder (200-300 MB)

```bash
mkdir -p gdrive_primary/outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/
mkdir -p gdrive_primary/config/model/

# Copy best checkpoint
cp outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt \
   gdrive_primary/outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/

# Copy configs
cp config/*.yaml gdrive_primary/config/
cp config/model/*.yaml gdrive_primary/config/model/
```

**Then:**
1. Upload `gdrive_primary/` to Google Drive as new folder "flow-matching-mnist-primary"
2. Get folder ID from URL
3. Update PRIMARY_ID in downloads.sh

### SECONDARY Folder (1-2 GB)

```bash
mkdir -p gdrive_secondary/outputs/fm/2026-05-01/22-58-12/
mkdir -p gdrive_secondary/results/

# Copy all checkpoints
cp -r outputs/fm/2026-05-01/22-58-12/training_logs/ \
      gdrive_secondary/outputs/fm/2026-05-01/22-58-12/

# Copy frame sequences
cp -r results/lda_frames/ gdrive_secondary/results/
```

**Then:**
1. Upload `gdrive_secondary/` to Google Drive as new folder "flow-matching-mnist-secondary"
2. Get folder ID from URL
3. Update SECONDARY_ID in downloads.sh
4. Commit updated downloads.sh to GitHub

---

## Post-Push Verification

- [ ] GitHub repo contains all code files
- [ ] GitHub repo contains documentation
- [ ] GitHub repo has .gitignore (excludes large files)
- [ ] downloads.sh has correct PRIMARY_ID
- [ ] downloads.sh has correct SECONDARY_ID
- [ ] Test: Clone from GitHub → run downloads.sh → everything works
- [ ] PRIMARY folder on Google Drive (200-300 MB)
- [ ] SECONDARY folder on Google Drive (1-2 GB)

---

## Final Steps

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Update downloads.sh with Google Drive IDs**
   ```bash
   # Edit downloads.sh
   PRIMARY_ID="xxxxx"
   SECONDARY_ID="xxxxx"
   
   git add downloads.sh
   git commit -m "Update: Set Google Drive folder IDs"
   git push origin main
   ```

3. **Create GitHub Release** (optional)
   ```bash
   git tag -a v1.0 -m "Initial release: Flow Matching MNIST"
   git push origin v1.0
   ```

---

**Ready for deployment!**  
**Last Updated:** 2026-05-02
