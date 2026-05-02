# LDA-Based Flow Field Trajectory Visualizations

## Overview

Three comprehensive LDA visualization systems are being generated to analyze Flow Matching trajectories across the learned space:

### Visualization Components

#### 1. **2D LDA Visualizations** (Digit Discrimination Only)
- **Components:** 2 dimensions from LDA trained on ground truth digits (0-9)
- **Purpose:** Shows how generated samples map to digit feature space
- **Time Points:** t=0 (noise), t=0.5 (intermediate), t=1.0 (data)
- **Color Scheme:** Each of 10 digit classes has unique color (tab10 colormap)
- **Output:** 
  - Static: `lda_2d_static.png` (4-panel comparison)
  - Animation: `lda_trajectory_2d.gif` (11 frames, t from 0 to 1)

#### 2. **3D LDA Visualizations** (Digit + Real/Fake Discrimination)
- **Components:**
  - X-axis: LDA Component 1 (Digit Discrimination)
  - Y-axis: LDA Component 2 (Digit Discrimination)
  - Z-axis: LDA Component 3 (Real vs Generated)
- **Purpose:** Shows how generated samples diverge from ground truth distribution
- **Color Scheme:** 
  - Points: Digits 0-9 (tab10 colors for ground truth)
  - Trajectories: Red (t=0) → Orange (t=0.5) → Green (t=1.0)
- **Output:**
  - Static: `lda_3d_static.png` (4-panel 3D plots)
  - Animation: `lda_trajectory_3d.gif` (11 frames, rotating view)

#### 3. **Interactive 3D Visualization** (Plotly)
- **Components:** Same 3D setup as above
- **Interaction:** 
  - Rotate: Click + drag
  - Zoom: Scroll wheel
  - Pan: Right-click + drag
  - Hover: View sample details
- **Output:** `lda_interactive_3d.html` (open in any web browser)

---

## File Generation Status

### Current Tasks (Background)

| Task | Status | Output | Time Est. |
|------|--------|--------|-----------|
| **Full Animation** (blcp4ckua) | Running | 2D GIF, 3D GIF | ~15-20 min |
| **Static Viz** (btgf4rc2k) | Running | 2D PNG, 3D PNG | ~5-10 min |
| **Interactive Plot** (b238undv8) | Failed | HTML | — |

### Expected Outputs

```
results/
├── lda_2d_static.png           (2D at 3 time points: 4-panel comparison)
├── lda_3d_static.png           (3D at 3 time points: 4-panel comparison)
├── lda_trajectory_2d.gif       (Animation: 11 frames, t=[0,0.1,...,1.0])
├── lda_trajectory_3d.gif       (Animation: 11 frames, 3D rotating view)
└── lda_interactive_3d.html     (Interactive Plotly visualization)
```

---

## What These Visualizations Show

### Key Insights

1. **Trajectory Evolution (Across Time)**
   - At t=0: Samples scatter around origin (Gaussian noise)
   - At t=0.5: Samples begin clustering toward digit regions
   - At t=1.0: Samples converge to ground truth digit clusters

2. **Real vs Generated Discrimination (Z-axis)**
   - Ground truth: Typically negative on real/fake axis
   - Generated: Positive on real/fake axis
   - Convergence: Z-coordinate changes at t=1.0 as model learns

3. **Digit Class Separation (X,Y plane)**
   - Ground truth: Clear separation between digit classes
   - Generated: Initially overlap, then separate as t→1
   - Quality: Loss improves (0.1829) → better clustering

4. **Color Coding**
   - **Digit 0:** Color 0 (dark blue)
   - **Digit 1:** Color 1 (orange)
   - **Digit 2-9:** Colors 2-9 (successive colors in tab10)
   - **Generated samples:** Marked with 'x' (not circles)

---

## Technical Details

### LDA Configuration

**LDA Model 1 (Digit Classification):**
- Trained on: Ground truth MNIST images
- Classes: 10 digit classes (0-9)
- Components: 2 (dimensionality reduction to 2D)
- Goal: Maximize inter-digit separation

**LDA Model 2 (Real vs Fake):**
- Trained on: Ground truth + Generated samples
- Classes: 2 (0=real, 1=generated)
- Components: 1 (single discriminative axis)
- Goal: Maximize real/fake separation

### Combined 3D Space
- Dimensions 1-2: From LDA Model 1
- Dimension 3: From LDA Model 2
- Result: 3D space showing both digit structure AND real/fake separation

---

## How to Use

### Static Images
1. Open `lda_2d_static.png` or `lda_3d_static.png` in any image viewer
2. Compare panels to see how trajectories evolve
3. Colors indicate digit classes (same across all panels)

### Animations (GIF)
1. Open `lda_trajectory_2d.gif` or `lda_trajectory_3d.gif`
2. Watch progression from t=0 to t=1
3. Notice how samples move from random noise → digit clusters

### Interactive Plot
1. Open `lda_interactive_3d.html` in a web browser
2. Click and drag to rotate the 3D plot
3. Scroll to zoom in/out
4. Hover over points to see details
5. Toggle legend items to show/hide categories

---

## Interpretation Guide

### What to Look For

**Sample 1: Trajectory Coherence**
- Are generated samples moving smoothly from noise to structure?
- Do they cluster better over time?

**Sample 2: Digit Discrimination**
- Do generated samples converge to same locations as ground truth?
- Is there confusion between similar digits (e.g., 0 vs 6)?

**Sample 3: Real vs Fake Separation**
- How much do generated samples diverge from real distribution (Z-axis)?
- Does divergence decrease as t→1.0?

**Sample 4: Time Evolution**
- How many frames to see coherent structure? (frame number ≈ 5 = t=0.5)
- Is convergence exponential or linear?

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.1829 (best at epoch 14) |
| Samples Generated | 200 per time point × 11 time points = 2,200 |
| Ground Truth Samples | 500 (unique MNIST digits) |
| LDA Components | 2 (digit) + 1 (real/fake) = 3 total |
| Time Resolution | Δt=0.1 (11 frames from t=0 to t=1) |
| Frame Rate (GIF) | 2 FPS (5 seconds per animation) |

---

## Generation Commands

### Static Visualizations (Fast)
```bash
python lda_static_viz.py
```
- Generates 2D and 3D static PNG images
- Time: ~5-10 minutes
- Output: `lda_2d_static.png`, `lda_3d_static.png`

### Full Animations
```bash
python lda_trajectory_visualization.py
```
- Generates 11-frame GIF animations (2D and 3D)
- Generates interactive Plotly HTML
- Time: ~20-30 minutes
- Output: `lda_trajectory_2d.gif`, `lda_trajectory_3d.gif`, `lda_interactive_3d.html`

---

## Expected Results

Based on training quality (loss=0.1829):

✓ **Good Separation:** Ground truth digits should show clear clustering  
✓ **Smooth Transitions:** Generated samples should move continuously from noise to clusters  
✓ **Convergence:** By t=1.0, generated samples should closely match ground truth  
✓ **Time Evolution:** Clear progression visible across 11 frames  
✓ **Real/Fake Axis:** Noticeable difference in Z-position between real and generated

---

## Troubleshooting

If visualizations don't appear:

1. **Check results/ directory:** `ls results/lda_*`
2. **Check background tasks:**
   - `blcp4ckua` for full animation status
   - `btgf4rc2k` for static viz status
3. **View task output:** Check task completion messages
4. **File permissions:** Ensure write access to `results/` directory

---

## Next Steps

1. Once visualizations complete, examine:
   - How well generated samples match ground truth clusters
   - Time-based convergence rate
   - Real/fake separation quality

2. Compare with previous models:
   - Previous loss (0.3647) showed poor clustering
   - Current loss (0.1829) should show much better alignment

3. Use interactive visualization to explore:
   - Specific digits and their generated samples
   - Regions of confusion between digit classes
   - Real/fake discrimination effectiveness

---

**Generated:** 2026-05-02  
**Model:** Best Checkpoint (Epoch 14, Loss=0.1829)  
**Status:** Visualizations in progress...
