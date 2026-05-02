# Complete Visualization Guide

This guide documents all visualizations available in the Flow Matching MNIST project, organized by category.

---

## Quick Start - Generate All Visualizations

```bash
pip install -r requirements.txt

# Sample Quality (5 min)
python visualizations/training/generate_samples.py

# Training Analysis (2 min)
python visualizations/training/analyze_training.py

# Interactive 3D (5 min)
python visualizations/lda/quick_3d_interactive.py

# LDA Frame Sequences (30-40 min)
python visualizations/lda/lda_frame_sequence.py
```

---

## Visualization Categories

### 1. Sample Quality Visualizations

#### Generated Samples Grid
- **Script:** `visualizations/training/generate_samples.py`
- **Output:** `results/generated_samples.png` (45 KB)
- **What:** 4×4 grid of 16 MNIST digits generated from the model
- **Good:** All digits are clear, diverse, recognizable
- **Bad:** Blurry, illegible, or all look the same

#### Quality vs ODE Steps
- **Script:** `visualizations/training/generate_samples.py`
- **Output:** `results/generation_process.png` (39 KB)
- **What:** 4 rows showing quality improvement with more ODE steps
- **Rows:** 5 steps, 10 steps, 20 steps, 50 steps
- **Trade-off:** More steps = cleaner but slower

#### Trajectory Evolution
- **Script:** `visualizations/training/generate_samples.py`
- **Output:** `results/trajectory_evolution.png` (27 KB)
- **What:** Single sample evolving from noise to digit at 4 time points
- **Shows:** How noise progressively becomes coherent digit

#### Training Summary
- **Script:** `visualizations/training/generate_samples.py`
- **Output:** `results/training_summary.png` (117 KB)
- **What:** 4-panel overview with samples, comparison, and stats

---

### 2. Training Analysis Visualizations

#### Loss Curves
- **Script:** `visualizations/training/analyze_training.py`
- **Output:** `results/loss_curves.png` (63 KB)
- **Content:** 2 panels
  - Left: Loss over epochs (exponential decay then plateau)
  - Right: Improvement % per epoch (51% total)
- **Key metrics:**
  - Initial: 0.3647 (Epoch 0)
  - Best: 0.1829 (Epoch 14)
  - Final: 0.1807 (Epoch 23)

#### Training Report
- **Script:** `visualizations/training/analyze_training.py`
- **Output:** `results/training_report.png` (359 KB)
- **What:** Comprehensive technical report with all details

---

### 3. Interactive 3D Visualization

#### 3D LDA Interactive Plot
- **Script:** `visualizations/lda/quick_3d_interactive.py`
- **Output:** `results/lda_3d_interactive_quick.html` (4.7 MB)
- **Time:** 5 minutes
- **Features:**
  - **Rotate:** Click + drag
  - **Zoom:** Scroll wheel
  - **Pan:** Right-click + drag
  - **Hover:** See sample details
  - **Toggle:** Click legend to show/hide digit classes

**What it shows:**
- X-axis: LDA Component 1 (digit discrimination)
- Y-axis: LDA Component 2 (digit discrimination)
- Z-axis: LDA Component 3 (real vs generated discrimination)
- 300 ground truth samples (colored by digit 0-9)
- 300 generated samples (red X marks)

**Interpretation:**
- Good model: Generated samples cluster near same-digit circles
- Bad model: Generated samples scattered randomly

---

### 4. LDA Trajectory Frame Sequences

#### 51-Frame 2D Sequence
- **Script:** `visualizations/lda/lda_frame_sequence.py`
- **Output:** 51 PNG files in `results/lda_frames/frame_2d_00.png` to `frame_2d_50.png`
- **Time:** 30-40 minutes
- **Size:** ~5 MB total
- **What:** 2D trajectory evolution over time
  - X-axis: LDA Component 1
  - Y-axis: LDA Component 2
  - Shows digit discrimination only

**Frame progression:**
- Frame 0 (t=0): Noise scattered around origin
- Frame 10 (t=0.2): Slight structure
- Frame 25 (t=0.5): Clear clustering
- Frame 50 (t=1.0): Converged to ground truth clusters

#### 51-Frame 3D Sequence
- **Script:** `visualizations/lda/lda_frame_sequence.py`
- **Output:** 51 PNG files in `results/lda_frames/frame_3d_00.png` to `frame_3d_50.png`
- **Size:** ~10 MB total
- **What:** 3D trajectory with real/fake discrimination
  - X,Y axes: Digit discrimination (same as 2D)
  - Z-axis: Real vs generated

**Z-axis interpretation:**
- Ground truth: Negative Z (labeled "real")
- Generated early frames: Positive Z (labeled "fake")
- Generated late frames: Negative Z (learned to be "real")

---

## Directory Structure

```
visualizations/
├── training/                 # Sample quality and training analysis
│   ├── generate_samples.py  # Generates 4 visualizations
│   └── analyze_training.py  # Loss curves and report
└── lda/                      # LDA-based trajectory analysis
    ├── quick_3d_interactive.py     # Interactive 3D (5 min)
    ├── lda_frame_sequence.py       # 51 frames (30-40 min)
    ├── lda_trajectory_visualization.py  # Full animation (WIP)
    └── [other LDA scripts]

results/
├── generated_samples.png            # Sample grid
├── generation_process.png          # Quality vs steps
├── trajectory_evolution.png        # Sample evolution
├── loss_curves.png                 # Training metrics
├── training_summary.png            # Overview
├── training_report.png             # Detailed report
├── lda_3d_interactive_quick.html   # Interactive 3D
└── lda_frames/                     # 102 frame images
    ├── frame_2d_00.png to frame_2d_50.png
    └── frame_3d_00.png to frame_3d_50.png
```

---

## Performance & Time Estimates

| Visualization | Time | Size | Notes |
|---------------|------|------|-------|
| Sample Grid | <1 min | 45 KB | Very fast |
| Quality Comparison | <1 min | 39 KB | Very fast |
| Loss Curves | <1 min | 63 KB | Reads existing logs |
| Training Report | <1 min | 359 KB | Reads existing logs |
| Interactive 3D | 5 min | 4.7 MB | Generates 300 samples |
| 2D Frames | 20 min | 5 MB | Generates 51 frames |
| 3D Frames | 20 min | 10 MB | Generates 51 frames |
| **Total** | **~45 min** | **~20 MB** | **All visualizations** |

---

## Viewing the Visualizations

### Static Images
```bash
# View any PNG
open results/generated_samples.png          # macOS
xdg-open results/generated_samples.png      # Linux
start results/generated_samples.png         # Windows
```

### Interactive 3D
```bash
# Open HTML in browser
open results/lda_3d_interactive_quick.html  # macOS
firefox results/lda_3d_interactive_quick.html  # Linux
start results/lda_3d_interactive_quick.html # Windows
```

### Frame Sequences
```bash
# Browse PNGs
ls results/lda_frames/frame_2d_*.png

# Create animated GIF (requires imagemagick)
convert -delay 50 results/lda_frames/frame_2d_*.png trajectory_2d.gif

# Create video (requires ffmpeg)
ffmpeg -framerate 2 -i "results/lda_frames/frame_2d_%02d.png" trajectory_2d.mp4
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model checkpoint not found | Run `bash scripts/downloads.sh` |
| Out of memory | Reduce `num_samples` parameter in scripts |
| Slow sample generation | Reduce ODE steps (use 20 instead of 50) |
| HTML file won't load | Use Chrome or Firefox (not IE) |
| Frame generation very slow | Normal - takes 1-2 sec per frame |
| Missing dependencies | Run `pip install -r requirements.txt` |

---

## Future Enhancements

Planned improvements for next version:

- [ ] **Trajectory Arrows:** Add velocity vectors to show movement direction
- [ ] **Dynamic LDA:** Refit basis per frame to show basis rotation
- [ ] **Animated GIFs:** Auto-convert frame sequences to MP4/WebM
- [ ] **Real-time Dashboard:** Training progress UI
- [ ] **Conditional Generation:** Class-conditional digit generation

---

**Status:** ✓ All visualizations ready for deployment  
**Last Updated:** 2026-05-02
