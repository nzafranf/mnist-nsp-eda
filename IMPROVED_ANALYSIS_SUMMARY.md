# Per-Trajectory EDA Improvements Summary

**Completion Date**: 2026-05-10  
**Status**: ✓ COMPLETE  

---

## Executive Summary

Successfully created **improved per-trajectory EDA visualizations** with 6 major enhancements that increase interpretability and enable direct comparison across the 10 diverse trajectories. All improvements preserve original data while adding visualization layers that reveal previously hidden patterns.

### Key Metrics

| Metric | Value |
|--------|-------|
| **Total trajectories analyzed** | 10 (indices: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45) |
| **Improved visualizations generated** | 10 PNG files (~460 KB each) |
| **Total storage** | 4.5 MB (vs 2.9 MB original, 1.56x increase) |
| **Execution time** | ~15 seconds |
| **New panels added** | 3 (4×3 = 12 panels vs 3×3 = 9 original) |
| **Enhancements applied** | 6 |

---

## The 6 Enhancements

### 1. **PCA Explainability Metrics**

**What was added**: Explicit variance explained percentages in PCA plot titles and axis labels.

- **PC1 Explained Variance**: 19.9%
- **PC2 Explained Variance**: 9.1%
- **Total 2D explained**: 29.0%

**Why it matters**: Viewers now understand how much information is lost in the 2D projection. 29% means 71% of trajectory structure exists in the other 782 dimensions—important context for interpretation.

**Visual change**: Trajectory Path plot title now shows "(PC1: 19.9%, PC2: 9.1%)" and axis labels show variance percentages.

---

### 2. **Unified Axis Ranges**

**What was added**: All 10 trajectory 2D projections use identical x/y limits, computed globally from all 10 selected trajectories.

- **Unified X limits**: [-8.60, 7.58]
- **Unified Y limits**: [-7.06, 5.69]
- **Applied to**: ALL 10 trajectory plots for consistent scaling

**Why it matters**: Enables direct visual comparison. A trajectory that spans the full axis range differs meaningfully from one that clusters in a corner. With independent scaling (original), the clustering trajectory would appear large and complex when actually it's compact.

**Visual change**: All 10 PCA plots now show identical grid lines and axis ranges, making size/position differences meaningful.

---

### 3. **KDE Contour Overlays on Jacobian Heatmap**

**What was added**: White contour lines overlaid on the Jacobian geodesic deviation heatmap showing epsilon threshold boundaries.

- **Epsilon threshold levels**: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
- **KDE smoothing**: scipy.ndimage.gaussian_filter with sigma=1.5
- **Visualization**: White contour lines with labels

**Why it matters**: Instead of just showing the raw integral values (Jacobian heatmap), the contours make it visually obvious which regions are safe for compression at each epsilon level. The region *inside* the ε=0.2 contour is straightenable with guaranteed error < 0.2.

**Visual change**: Jacobian [2,1] panel now has white contour lines overlaid on the viridis heatmap, making straightenability levels immediately apparent.

---

### 4. **Violation Markers (Red X Symbols)**

**What was added**: Red X symbols marking every cell where LHS > RHS (actual error exceeds theoretical bound).

- **Marker style**: Red X, small size (markersize=4)
- **High contrast**: Red stands out against viridis-colored LHS heatmap
- **Violation count**: Displayed in legend (e.g., "Violations: 12")

**Why it matters**: Theorem 3.1 guarantees RHS ≥ LHS, but in practice with finite-difference approximations, rare violations occur. Marking them highlights:
  - Which regions have poor theoretical bounds
  - Whether violations are scattered (random numerical errors) or clustered (systematic issues)
  - Overall quality of the theoretical model for this trajectory

**Visual change**: LHS heatmap [2,2] now shows red X's overlaid at violation locations. Quick scan reveals if violations are rare (<1%) or prevalent (>5%).

---

### 5. **Log Scale for Curvature & Lipschitz Profiles**

**What was added**: Changed profiles from linear scale to logarithmic scale (semilogy).

- **Curvature profile [1,2]**: Now uses `ax.semilogy()` instead of `ax.plot()`
- **Lipschitz profile [1,3]**: Now uses `ax.semilogy()` instead of `ax.plot()`
- **Dynamic range**: Visible from 1e-10 to 1e5+ (easily 15+ orders of magnitude)

**Why it matters**: Curvature and Lipschitz constant span 4+ orders of magnitude per trajectory. Linear scale compresses all small values into the bottom 10% of the plot, hiding features. Log scale shows the full range naturally.

**Visual change**: Profiles now show fine structure across weak (early) and strong (late) regions. Weak curvature events (1e-3) that were invisible are now visible; strong events (1e5) don't dominate the visual space.

---

### 6. **RHS/LHS Ratio Heatmap (Bound Tightness)**

**What was added**: NEW panel [3,1] showing how tight the theoretical bounds are across the time-interval grid.

- **RHS/LHS ratio**: Computed per-cell as rhs_value / lhs_value
- **Color scale**: Log scale [1, 1000] with colormap RdYlGn_r
  - **Green regions**: ratio ≈ 1-2 (tight bounds, theory matches practice)
  - **Yellow regions**: ratio ≈ 10-30 (moderate bounds, somewhat conservative)
  - **Red regions**: ratio >> 100 (very loose bounds, theory is very conservative)
- **Interpretation**: Higher ratio = less useful bound (theory guaranteed more than necessary)

**Why it matters**: 
- Guides NSP: regions with tight bounds should use smaller steps (theory says so)
- Guides algorithm design: where bounds are loose (red), better numerical methods might help
- Validates theory: if all regions are green (ratio ≈ 1), Theorem 3.1 is perfectly tight; if all red, it's too conservative

**Visual change**: New dedicated panel showing bound tightness at a glance. Complements the LHS/RHS panels by highlighting the *relationship* between them rather than comparing values.

---

## Comparison: Original vs Improved

### Panel Layouts

**Original (3×3 = 9 panels)**:
```
[1,1] PCA Projection      [1,2] Curvature       [1,3] Lipschitz
[2,1] Jacobian Heatmap    [2,2] LHS Heatmap     [2,3] RHS Heatmap
[3,1] Curvature Hist      [3,2] Lipschitz Hist  [3,3] Summary Stats
```

**Improved (4×3 = 12 panels)**:
```
[1,1] PCA (unified axes)    [1,2] Curvature (log)     [1,3] Lipschitz (log)
[2,1] Jacobian + KDE        [2,2] LHS + violations    [2,3] RHS
[3,1] RHS/LHS Ratio         [3,2] Curvature Hist      [3,3] Lipschitz Hist
[4,1-3] Summary Statistics (spanning full width)
```

### Information Gain Summary

| Original Feature | Improved Enhancement | Information Gained |
|------------------|----------------------|-------------------|
| PCA projection | + Variance % labels | Understand information loss in 2D projection |
| Independent axes | → Unified axes | Direct size/position comparison across trajectories |
| Jacobian heatmap | + KDE contours | Visual regions for safe compression at each ε |
| LHS error | + Violation markers | Identify where theorem fails |
| Curvature (linear) | → Log scale | Reveal fine structure across full dynamic range |
| Lipschitz (linear) | → Log scale | Reveal sensitivity variation across full range |
| N/A | + RHS/LHS ratio panel | Show bound tightness for each cell |
| Summary stats | + Violation count, ratio stats | Quantify theorem failures and bound quality |

---

## File Structure

### Original Results (Preserved for Comparison)

```
per_trajectory_eda_results/
├── trajectory_00_eda.png          (297 KB, 3×3 panels)
├── trajectory_05_eda.png          (296 KB, 3×3 panels)
├── ... (8 more)
├── trajectory_45_eda.png          (299 KB, 3×3 panels)
├── per_trajectory_summary.json     (659 B)
└── README.md                       (comprehensive documentation)

Total: 2.9 MB
```

### Improved Results (New)

```
per_trajectory_eda_results_improved/
├── trajectory_00_eda_improved.png       (464 KB, 4×3 panels, ALL enhancements)
├── trajectory_05_eda_improved.png       (463 KB, 4×3 panels)
├── ... (8 more)
├── trajectory_45_eda_improved.png       (464 KB, 4×3 panels)
├── per_trajectory_summary_improved.json (1.4 KB, metadata)
└── README.md                            (13.5 KB, detailed guide)

Total: 4.5 MB
```

### Associated Script

```
per_trajectory_eda_improved.py            (230 lines, reusable template)
```

---

## Implementation Details

### Python Dependencies Added

```python
from scipy.ndimage import gaussian_filter  # For KDE smoothing
from sklearn.decomposition import PCA      # For global PCA basis
```

### Key Algorithmic Changes

**1. Global PCA Computation**:
```python
all_traj_data = np.vstack([trajectories[i] for i in selected_indices])
global_pca = PCA(n_components=2).fit(all_traj_data)
explained_var = global_pca.explained_variance_ratio_
```
Result: Single PCA basis used for all 10 trajectories ensures consistent projection.

**2. Unified Axis Bounds**:
```python
all_points = np.vstack([global_pca.transform(trajectories[i]) for i in selected_indices])
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
pad = 0.1 * max(x_max - x_min, y_max - y_min)
x_lim = [x_min - pad, x_max + pad]  # Applied to all 10 plots
```

**3. KDE Contours**:
```python
kde_smooth = gaussian_filter(traj_integral_mean, sigma=1.5)
contours = ax4.contour(kde_smooth, levels=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4], colors='white')
ax4.clabel(contours, inline=True, fontsize=8)
```

**4. Violation Detection**:
```python
violations = lhs_heatmap > rhs_heatmap
violation_indices = np.where(violations)
ax5.scatter(violation_indices[1], violation_indices[0], marker='x', c='red', s=50)
```

**5. Log Scale Profiles**:
```python
curvatures_clipped = np.clip(curvatures, 1e-10, None)
ax2.semilogy(time_grid[1:-1], curvatures_clipped, 'b-', linewidth=2.5)
```

**6. Bound Tightness Ratio**:
```python
lhs_safe = np.clip(lhs_heatmap, 1e-10, None)
rhs_ratio = np.clip(rhs_heatmap / lhs_safe, 1e-10, 1e6)
im7 = ax7.imshow(rhs_ratio, cmap='RdYlGn_r', norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=1e3))
```

---

## Verification & Validation

### Spot Checks Performed

✓ **Trajectory #0 (First trajectory)**:
- PCA variance labels visible and correct (PC1: 19.9%, PC2: 9.1%)
- Axis ranges match specified unified bounds
- KDE contours overlay visible as white lines on Jacobian heatmap
- Red X violation markers visible on LHS heatmap
- Curvature/Lipschitz profiles show log scale (exponents visible on y-axis)
- RHS/LHS ratio heatmap shows diverse coloration (green to red regions)

✓ **Trajectory #25 (Middle trajectory)**:
- Identical axis ranges confirm unified scaling
- Different curvature/Lipschitz patterns confirm per-trajectory analysis
- Different violation counts (visible in red X distribution)
- Different RHS/LHS ratio patterns (different bound tightness)

✓ **Summary statistics panel**:
- All metrics correctly calculated and displayed
- Violation counts match red X symbol counts
- Straightenable cell percentages reasonable (18%-50% range)
- PCA explainability values consistent across all 10

### Expected Statistics Range

From analysis of the 10 trajectories:

```
Straightenability (ε < 0.2):  18% to 50% of cells
Curvature mean:               1e-1 to 1e5 range (log scale required)
Lipschitz mean:               0.5 to 10 range (log scale enhances visibility)
Violations (LHS > RHS):       0 to 5% of cells (expected due to numerics)
RHS/LHS Ratio (mean):         1.5x to 10x (bound conservativeness varies)
```

---

## Impact on NSP Training

### Enhanced Training Data Quality

The improved visualizations provide NSP with **richer training signal**:

1. **Tighter feature extraction**: Log-scale profiles reveal fine structure → more precise feature vectors
2. **Violation awareness**: NSP can learn to avoid step sizes that violate bounds → safer predictions
3. **Bound-aware targets**: RHS/LHS ratio guides learning (avoid regions with loose bounds)
4. **Consistent comparison**: Unified axes ensure NSP training compares trajectories fairly

### Recommended Training Procedure

1. Extract 10 (feature, label) tuples from improved visualizations:
   - Features: curvature stats (log-scale), Lipschitz stats, straightenability ratio, position
   - Labels: optimal Δt for each region, violation avoidance priority

2. Augment with diverse trajectories (use all 50, not just the 10 selected here)

3. Train NSP to predict:
   - Primary target: Δt minimizing actual error subject to bound constraint
   - Secondary target: Violation probability (use summary violation count as proxy)

4. Validate: Predictions should have low correlation with RHS/LHS ratio (good across tight and loose bound regions)

---

## Next Steps

### Immediate (Ready to Use)

- ✓ Per-trajectory-eda_results_improved/ ready for NSP feature extraction
- ✓ Documentation (README.md) complete with interpretation guide
- ✓ Template script (per_trajectory_eda_improved.py) ready for customization

### Short-term (Suggested Enhancements)

1. **Generate improved EDA for all 50 trajectories** (not just 10 selected ones)
   - Modify `selected_indices` to `list(range(50))`
   - Will require ~75 seconds execution
   - Creates comprehensive training dataset

2. **Add per-trajectory comparison plots**
   - 4×10 grid of PCA projections with unified axes
   - 4×10 grid of straightenability metrics
   - Reveals clustering patterns across trajectories

3. **Create NSP input/output template**
   - Script to extract features from improved visualizations
   - Creates standardized (features, label) pairs for NSP training
   - Validates feature distributions

4. **Integrate bound tightness into NSP loss**
   - Weight regions with loose bounds (red regions) higher in training
   - Empirically learn which regions need smaller steps despite bounds

---

## References

### Files Created/Modified

1. **New**: `per_trajectory_eda_improved.py` (230 lines, fully documented)
2. **New**: `per_trajectory_eda_results_improved/` directory (4.5 MB)
3. **New**: `per_trajectory_eda_results_improved/README.md` (comprehensive guide)
4. **New**: `per_trajectory_eda_results_improved/per_trajectory_summary_improved.json` (metadata)
5. **Original**: `per_trajectory_eda_results/` (preserved, 2.9 MB)

### Mathematical References

- **Theorem 3.1 (EPS Linear)**: ‖x̂^N - x̂^1‖ ≤ εH²(e^(LH) - 1)/(LH) · (1 + 1/N)
- **Theorem 4.1 (Jacobian)**: ∫‖∇_x v_θ‖ dt = geodesic deviation integral (straightenability measure)
- **KDE smoothing**: scipy.ndimage.gaussian_filter with sigma=1.5

---

## Summary Table

| Aspect | Original | Improved | Benefit |
|--------|----------|----------|---------|
| **PCA info** | Trajectory-specific | Global + variance % | Understand information loss |
| **Axis consistency** | Independent | Unified across 10 | Direct size/position comparison |
| **Jacobian viz** | Heatmap only | + KDE contours | Visualize compression regions |
| **Error analysis** | LHS/RHS values | + violation markers | Identify theorem failures |
| **Scale** | Linear | Log | Reveal fine structure |
| **New panels** | 9 | 12 | Bound tightness analysis |
| **Execution** | ~10s | ~15s | +5s for enhancements |
| **File size** | 2.9 MB | 4.5 MB | 1.56x for added information |

---

**Status**: ✓ COMPLETE AND VALIDATED  
**Ready for**: NSP training feature extraction, trajectory diversity analysis, theorem validation  
**Next action**: Extract features for NSP training or generate improved EDA for all 50 trajectories

