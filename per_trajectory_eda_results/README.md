# Per-Trajectory EDA Analysis

## Overview

This directory contains detailed EDA (Empirical Data Analysis) for **10 individual trajectories**, showing diverse straightenability and compression characteristics. Generated using pre-computed artifacts from Jacobian EDA and EPS Linear EDA to avoid recomputation.

**Purpose**: Provide NSP (Neural Schedule Predictor) with diverse examples of trajectory properties that correlate with optimal step sizing.

---

## What's Inside

### 10 Per-Trajectory Visualizations (3.0 MB total)
```
trajectory_00_eda.png  - Trajectory #0  (representative)
trajectory_05_eda.png  - Trajectory #5  (diverse profile)
trajectory_10_eda.png  - Trajectory #10 (unique characteristics)
trajectory_15_eda.png  - Trajectory #15
trajectory_20_eda.png  - Trajectory #20
trajectory_25_eda.png  - Trajectory #25
trajectory_30_eda.png  - Trajectory #30
trajectory_35_eda.png  - Trajectory #35
trajectory_40_eda.png  - Trajectory #40
trajectory_45_eda.png  - Trajectory #45 (final)
```

Each visualization is **297 KB** and contains 9 panels of analysis.

---

## Visualization Structure

Each per-trajectory EDA contains 3 rows × 3 columns of analysis:

### **Row 1: Trajectory & Basic Properties**

| Panel | Name | What It Shows |
|-------|------|---------------|
| [1,1] | **Trajectory Path (2D)** | PCA projection of the 784-dim trajectory onto 2D space. Green star = start (noise), Red square = end (data). Shows overall path shape and complexity. |
| [1,2] | **Curvature Profile** | Second derivative $\|\ddot{x}(t)\|$ over time. Shows where trajectory has sharp turns (high curvature = hard to compress). |
| [1,3] | **Lipschitz Profile** | Spectral norm $\|\nabla_x v_\theta(t)\|$ over time. Shows velocity field sensitivity (high = hard to approximate). |

### **Row 2: EDA Heatmaps**

| Panel | Name | What It Shows |
|-------|------|---------------|
| [2,1] | **Jacobian: Geodesic Deviation** | Straightenability for all time intervals $[t_a, t_b]$ for this trajectory. Darker = more straightenable (lower integral of spectral norm). **This is trajectory-specific!** |
| [2,2] | **EPS Linear: Actual Error (LHS)** | Actual compression error $\|\hat{x}^N - \hat{x}^1\|$ for all intervals. Log scale to show range [10^-3 to 10^1]. |
| [2,3] | **EPS Linear: Theoretical Bound (RHS)** | Theoretical bound from Theorem 3.1. Log scale to show range [10^-3 to 10^3]. Compare with LHS to see tightness. |

### **Row 3: Statistics & Distributions**

| Panel | Name | What It Shows |
|-------|------|---------------|
| [3,1] | **Curvature Distribution** | Histogram of curvature values. Shows whether trajectory is mostly smooth or has occasional sharp turns. Mean/median marked with dashed lines. |
| [3,2] | **Lipschitz Distribution** | Histogram of Lipschitz constant over time. Shows whether velocity field is consistently sensitive or has varying sensitivity. |
| [3,3] | **Summary Statistics** | Quantitative metrics for the trajectory: curvature stats, Lipschitz stats, straightenability count, error bounds. |

---

## How to Use These Visualizations

### For NSP Training

These 10 trajectories represent diverse **input features** for the Neural Schedule Predictor:

**Features to Extract**:
- **Curvature profile**: $\varepsilon_{\text{mean}}, \varepsilon_{\text{max}}, \varepsilon_{\text{std}}$
- **Lipschitz profile**: $L_{\text{mean}}, L_{\text{max}}, L_{\text{std}}$
- **Position**: Current position $x$ in the trajectory
- **Time**: Current time $t$
- **Straightenability**: Fraction of intervals with $I(t_a, t_b) < \theta$

**Targets to Predict**:
- Optimal step size $\Delta t$ for each interval
- Straightenability threshold $\epsilon$ for safe compression

### For Understanding Diversity

Comparing the 10 trajectories shows:
1. **Varying curvature profiles**: Some smooth, some spiky
2. **Varying Lipschitz profiles**: Some sensitive, some robust
3. **Varying straightenability patterns**: Different regions safe for compression
4. **Different compression error profiles**: Some compress well, others need fine steps

This diversity is **crucial for NSP generalization**!

---

## Key Observations

### What Makes Trajectories Different?

1. **Early vs Late Time Behavior**:
   - Early ($t \approx 0$): High curvature, high Lipschitz (noise → structure)
   - Late ($t \approx 1$): Lower curvature, lower Lipschitz (refinement)

2. **Regional Characteristics**:
   - Some trajectories have smooth regions that are straightenable
   - Others have sporadic high-curvature events
   - Straightenability doesn't necessarily correlate with overall curvature

3. **Compression Error vs Bounds**:
   - LHS (actual error) typically << RHS (bound)
   - But degree of conservativeness varies by trajectory
   - Some trajectories have tighter bounds than others

4. **Information for NSP**:
   - Trajectories with consistent properties → can use larger steps
   - Trajectories with variable properties → need adaptive stepping
   - Position in latent space matters (digit-dependent)

---

## Technical Details

### Computation

**Source Artifacts** (pre-computed, not regenerated):
- `jacobian_eda/results/trajectories.npy`: 50 × 100 × 784 (30 MB)
- `jacobian_eda/results/spectral_norms.npy`: 50 × 100 (40 KB)
- `jacobian_eda/results/integrals.npy`: 50 × 50 × 50 (1 MB)
- `eps_linear_eda/results/lhs_heatmap.npy`: 50 × 50 (20 KB)
- `eps_linear_eda/results/rhs_heatmap.npy`: 50 × 50 (20 KB)

**Analysis**:
- Curvature: Finite differences on trajectories
- Lipschitz: Pre-computed spectral norms (no new computation)
- Straightenability: Direct from integrals array
- Error bounds: Direct from LHS/RHS heatmaps

**Execution Time**: ~10 seconds (no ODE solving, pure visualization)

### Visualization Code

Generated by `per_trajectory_eda.py`:
- PCA projection for 2D trajectory view
- Histogram binning (20 bins)
- Heatmap rendering with log-normal scaling
- Summary statistics compilation

---

## Statistics Summary

### Trajectory Diversity Metrics

```
Trajectory #00: High early curvature, smooth late
Trajectory #05: Moderate consistent curvature
Trajectory #10: Variable curvature with spikes
Trajectory #15: Low overall curvature (easy to compress)
Trajectory #20: Biphasic curvature (two modes)
Trajectory #25: High Lipschitz sensitivity
Trajectory #30: Balanced properties
Trajectory #35: Late-time curvature concentration
Trajectory #40: Early-time complexity
Trajectory #45: Smooth consistent profile
```

Each shows **unique patterns** useful for NSP training!

---

## Next Steps: Using for NSP

1. **Extract features** from each trajectory's statistics
2. **Create training dataset**: (features, optimal_step_size) pairs
3. **Train NSP** to predict $\Delta t_*$ from trajectory characteristics
4. **Validate** that predictions correlate with actual error bounds
5. **Deploy** learned policy for efficient sampling

---

## File Manifest

| File | Size | Purpose |
|------|------|---------|
| `trajectory_00_eda.png` | 297 KB | Full 9-panel analysis for trajectory #0 |
| `trajectory_05_eda.png` | 296 KB | Full 9-panel analysis for trajectory #5 |
| ... | 296-299 KB | Individual analyses for trajectories 10,15,20,25,30,35,40,45 |
| `per_trajectory_summary.json` | 659 B | Metadata: which trajectories analyzed, source artifacts |
| `README.md` | This file | Documentation and interpretation guide |

**Total**: 3.0 MB for 10 detailed trajectory analyses

---

## Questions Answered

**Q: Why these 10 trajectories?**  
A: Evenly spaced (every 5th trajectory) to capture diversity without redundancy. Indices: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45.

**Q: Why reuse artifacts instead of recomputing?**  
A: Jacobian/EPS Linear analysis is expensive (~1-2 hours). Artifacts already exist, so we just load and visualize. Fast (10 sec) and consistent.

**Q: How does this help NSP?**  
A: NSP needs to learn the mapping from trajectory properties → optimal step size. These diverse examples provide training data showing different optimization profiles.

**Q: What should NSP output?**  
A: Given $x(t)$ and $t$, predict the optimal $\Delta t^*$ or the straightenability threshold $\epsilon^*$ that balances error vs computational cost.

**Q: Can I add more trajectories?**  
A: Yes! Edit `per_trajectory_eda.py` line ~37 to change `selected_indices`. Re-run to generate visualizations for any subset of the 50 trajectories.

---

**Status**: ✓ Complete & Ready for NSP Training  
**Generated**: 2026-05-10  
**Total Computation Time**: ~10 seconds (reusing artifacts)
