# Flow Matching Trajectory EDA
## Exploratory Data Analysis of FM Inference Trajectories on MNIST

A comprehensive first-principles analysis of Flow Matching (FM) trajectory geometry, curvature, information gain, and schedule optimization potential. This EDA grounds the Neural Schedule Predictor (NSP) concept with empirical observations.

---

## Overview

This project performs a systematic exploratory data analysis on FM inference trajectories using MNIST as the test domain. Rather than formal proofs, we focus on strong heuristic observations of:

- **Trajectory Geometry**: PCA projections, straightness metrics
- **Curvature & Alignment**: Correlation between path curvature and velocity direction
- **Information Gain**: Velocity magnitude profiles and phase transitions
- **Leap Viability**: How much timesteps can be skipped while maintaining accuracy
- **Schedule Optimization**: Greedy oracle schedules and speedup potential
- **Trajectory Clustering**: Whether trajectory types suggest class-conditional schedules

---

## Quick Start

### Option 1: Automated Pipeline (Recommended)

```bash
# Train a model and run full EDA (3 epochs + analysis)
python run_eda.py --num-trajectories 20 --num-epochs 3

# Or use an existing checkpoint
python run_eda.py --checkpoint path/to/checkpoint.ckpt --num-trajectories 20

# Or use quick training with more trajectories
python run_eda.py --num-epochs 5 --num-trajectories 50
```

### Option 2: Step-by-Step

```bash
# 1. Train a model (if needed)
python -m src.quick_train --epochs 5 --batch-size 32

# 2. Get the checkpoint path
ls -la checkpoints/

# 3. Run EDA on the checkpoint
python -m src.eda_master --checkpoint checkpoints/fm-best-*.ckpt --num-trajectories 20
```

---

## Architecture

### Module Structure

```
src/
├── trajectory_logger.py          # Log intermediate states during ODE integration
├── trajectory_analysis.py        # Core analysis routines (curvature, alignment, etc.)
├── instrumented_generation.py    # Capture trajectories with instrumented ODE solver
├── eda_master.py                 # Master orchestrator for all 9 EDA phases
└── quick_train.py                # Lightweight training script
```

### Key Classes

#### `TrajectoryLogger`
Captures intermediate states during ODE integration:
- Records state at each timestep
- Logs velocity field values
- Saves/loads NPZ format
- Extracts trajectory dictionaries

#### `TrajectoryAnalyzer`
Static methods for trajectory metrics:
- `compute_curvature_proxy()` - Velocity magnitude differences
- `compute_alignment()` - Dot product with target direction
- `compute_straightness()` - Chord vs arc length
- `evaluate_all_possible_leaps()` - Test skip feasibility
- `greedy_schedule_discovery()` - Find optimal timestep selection
- `extract_trajectory_features()` - Hand-crafted features for clustering

#### `VisualizationUtils`
Plotting routines:
- `visualize_trajectory_pca()` - 2D PCA projections
- `plot_leap_profile()` - Leap size vs timestep
- `plot_information_gain()` - Velocity magnitude + leap regions
- `plot_curvature_alignment()` - Curvature and alignment overlays

#### `EDAMaster`
Orchestrates all 9 analysis phases with reporting.

---

## The 9 EDA Phases

### **Phase 1: Repository Setup and Instrumentation**
**Output**: 20-50 trajectory files (NPZ format)

Creates instrumented ODE solver that logs:
- States at each integration step
- Velocity field values
- Timestep metadata

```
results/phase1/
├── trajectory_000.npz
├── trajectory_001.npz
├── ...
└── summary.json
```

**Key Metrics**:
- Number of trajectories generated
- Trajectory shapes
- ODE integration steps

---

### **Phase 2: Basic Trajectory Visualization**
**Output**: PCA projection plots

Projects high-dimensional trajectories to 2D/3D PCA space:
- Shows trajectory "shape" in lower dimensions
- Color-coded by timestep progression
- Reveals curvature patterns

```
results/phase2/
├── trajectory_pca_00.png
├── trajectory_pca_01.png
├── ...
└── summary.json
```

**Questions Answered**:
- Are trajectories straight or curved?
- Do they all have similar shapes?
- Are there sharp bending regions?

---

### **Phase 3: Curvature and Alignment Analysis**
**Output**: Curvature-alignment plots + correlation statistics

Computes:
- **Curvature Proxy**: ||v(t+dt) - v(t)|| / dt
- **Alignment Score**: cos(angle between velocity and target direction)
- **Straightness**: chord distance / arc length
- **Correlation**: Pearson r between curvature and alignment

```
results/phase3/
├── curvature_alignment_00.png
├── curvature_alignment_01.png
├── ...
└── summary.json
  ├── individual_trajectories[].mean_curvature
  ├── individual_trajectories[].mean_alignment
  ├── individual_trajectories[].straightness
  ├── correlations.curvature_alignment_corr  (-0.68 expected)
  └── correlations.p_value
```

**Expected Observations**:
- Strong negative correlation (r < -0.6, p < 0.001)
- High curvature → low alignment
- Straightness ~0.7 ± 0.1

---

### **Phase 4: Leap Viability Analysis**
**Output**: Leap profile plots showing maximum leap sizes

For each timestep, finds the longest valid "leap" (skip):
- **Leap Error**: ||leap_state - true_state|| using single Euler step
- **Valid Leap**: Error < ε AND alignment > 0.5
- **Leap Size**: Maximum number of timesteps to skip

```
results/phase4/
├── leap_profile_00.png
├── leap_profile_01.png
├── ...
└── summary.json
  └── trajectories[].mean_leap_size  (3-5 steps expected)
  └── trajectories[].max_leap_size
```

**Expected Pattern**:
- Early phase: small leaps (navigation)
- Middle phase: larger leaps
- Late phase: variable/smaller (refinement)

---

### **Phase 5: Information Gain Proxy**
**Output**: Velocity magnitude profiles + phase transition detection

Analyzes ||v(s_t)|| as information transfer proxy:
- **Phase Transition Detection**: Where 75th percentile velocity is exceeded
- **Navigation Phase**: Low-variance broad uncertainty movement
- **Refinement Phase**: High-variance detail adjustment

```
results/phase5/
├── information_gain_00.png
├── information_gain_01.png
├── ...
└── summary.json
  └── trajectories[].phase_transition_idx  (~40% expected)
  └── trajectories[].phase_transition_ratio
```

**Key Insight**: Trajectory can be split into two distinct phases with different characteristics.

---

### **Phase 6: Greedy Schedule Discovery**
**Output**: Schedule optimization statistics

Greedily finds schedules by taking longest valid leap at each step:
- Compare vs uniform schedules (K=4, 8, 16)
- Compute speedup vs uniform
- Show that greedy exploits trajectory geometry

```
results/phase6/
└── summary.json
  ├── trajectories[].greedy_steps    (~8 expected for 100-step trajectory)
  ├── trajectories[].uniform_k8_steps (8)
  └── trajectories[].speedup         (1.5-2.5x expected)
```

**Expected Result**: 1.5-2.5x speedup vs uniform sampling.

---

### **Phase 7: Clustering Analysis**
**Output**: Trajectory feature clustering + heatmaps

Extract features for each trajectory:
- Mean/max curvature
- Mean alignment
- Straightness
- Mean leap size
- Complexity (phase transition ratio)

K-means clustering (k=2-3):
- **Cluster 0**: Simple/straight trajectories
- **Cluster 1**: Complex/curved trajectories
- **Cluster 2**: Mixed

```
results/phase7/
└── summary.json
  ├── features[].mean_curvature
  ├── features[].mean_alignment
  ├── features[].straightness
  ├── features[].mean_leap_size
  ├── clusters[cluster_0].count
  ├── clusters[cluster_0].mean_straightness
  └── clusters[cluster_0].mean_leap_size
```

**Insight**: Trajectory geometry clusters suggest that class-conditional schedules could improve efficiency further.

---

### **Phase 8: Ablation Study**
**Output**: Schedule comparison statistics

Evaluates and compares:
1. **Uniform Schedules**: Fixed K∈{4, 8, 16}
2. **Greedy Oracle**: From Phase 6
3. **Random**: Random timestep selection

```
results/phase8/
└── summary.json
  ├── schedules[uniform_K4].steps
  ├── schedules[uniform_K8].steps
  ├── schedules[uniform_K16].steps
  ├── schedules[greedy].steps
  └── schedules[random].steps
```

Shows clear advantage of greedy scheduling.

---

### **Phase 9: Summary Report**
**Output**: Comprehensive text report with key observations

Synthesizes findings from all phases:

```
results/phase9/
└── summary_report.txt
```

**Report Structure**:
1. Trajectory statistics (count, straightness)
2. Curvature-alignment correlation
3. Leap feasibility summary
4. Phase transition timing
5. Greedy schedule speedup
6. Clustering insights
7. NSP design implications

**Example Report**:
```
=== EDA Summary ===
Total trajectories analyzed: 20
Average straightness: 0.72 ± 0.12
Curvature-alignment correlation: -0.68 (p < 0.001)
Phase transition occurs at timestep: ~40% into trajectory
Average leap size (greedy): 3.2 steps (±1.5)
Greedy speedup vs. uniform (K=16): 2.1x
Trajectory clusters: 3 groups identified
```

---

## Results Interpretation Guide

### Straightness (Phase 2-3)
- **0.8-1.0**: Nearly straight path (rare)
- **0.6-0.8**: Moderate curvature (typical)
- **< 0.6**: High curvature (complex trajectories)

### Curvature-Alignment Correlation (Phase 3)
- **r < -0.6, p < 0.001**: Strong evidence that high curvature ↔ low alignment
  - Interpretation: Curved paths point away from target
  - NSP implication: Velocity direction is learnable feature
  
- **-0.6 < r < -0.3**: Moderate relationship
- **|r| < 0.3**: Weak/no relationship (unexpected)

### Leap Sizes (Phase 4)
- **Mean leap ~2-3**: Conservative stepping (complex trajectories)
- **Mean leap ~5-8**: Aggressive stepping (simpler trajectories)
- **Leap size varies along trajectory**: Phase-dependent (navigation vs refinement)

### Phase Transition Timing (Phase 5)
- **~30%**: Early high-gain phase (fast evolution)
- **~40%**: Balanced transitions
- **~50%+**: Late high-gain phase (gradual evolution)

### Greedy Speedup (Phase 6)
- **1.5-2.5x**: Typical speedup vs uniform
- **>3x**: Exceptional (very exploitable trajectories)
- **<1.5x**: Limited room for optimization (less structured)

### Clustering (Phase 7)
- **2-3 clusters**: Sufficient trajectory diversity
- **> 4 clusters**: Highly diverse trajectories
- **Cluster separation**: Whether geometry alone predicts schedule efficiency

---

## File Formats

### Trajectory NPZ Format (Phase 1)
```python
# Loading trajectory
data = np.load('results/phase1/trajectory_000.npz')
states = data['states']        # (T, 1, 28, 28) - trajectory images
times = data['times']          # (T,) - timestep values
velocities = data['velocities'] # (T, 1, 28, 28) - velocity field
targets = data['targets']      # (T, 1, 28, 28) or None
```

### Summary JSON Format
Each phase produces `summary.json` with:
- Phase-specific metrics
- Scalar summaries
- File listings

Example:
```json
{
  "num_trajectories": 20,
  "mean_straightness": 0.72,
  "trajectories": [
    {
      "idx": 0,
      "mean_curvature": 0.45,
      "max_curvature": 0.89
    }
  ]
}
```

---

## Customization

### Analyzing More Trajectories
```bash
python run_eda.py --num-trajectories 100
```
More trajectories → more robust statistics, longer runtime.

### Different Training Duration
```bash
python run_eda.py --num-epochs 10 --num-trajectories 50
```
Longer training → better model → cleaner trajectory structure.

### Custom Epsilon (Leap Error Threshold)
Edit `src/trajectory_analysis.py`, line ~200:
```python
leap_data = TrajectoryAnalyzer.evaluate_all_possible_leaps(
    states, velocities, alignment, target, epsilon=0.75  # Increase for more leaps
)
```

### Different ODE Integration Steps
```bash
python run_eda.py --num-trajectories 20
# Then edit src/eda_master.py, Phase 1:
# Change num_steps=100 to desired value
```

---

## Expected Runtime

| Task | Duration | Hardware |
|------|----------|----------|
| Quick train (3 epochs) | 5-10 min | CPU/GPU |
| Trajectory generation (20 traj, 100 steps) | 2-5 min | GPU recommended |
| EDA Phase 1-4 | 2-3 min | CPU |
| EDA Phase 5-9 | 1-2 min | CPU |
| **Total** | **~15-20 min** | GPU recommended |

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in `quick_train.py`
- Reduce `num_trajectories`
- Use CPU: Device will auto-select

### "Missing checkpoint"
```bash
# Ensure training completes
python -m src.quick_train --epochs 5
# Then run EDA
python run_eda.py --checkpoint checkpoints/fm-best-*.ckpt
```

### "Trajectories look strange"
- Increase training epochs (model needs more data)
- Check normalization in models/fm.py (should be [-1, 1])

### Visualization not saving
- Check write permissions in `results/` directory
- Ensure `results/phase{1-9}/` directories exist

---

## Paper References

This EDA framework is designed to ground the **Neural Schedule Predictor (NSP)** concept:
- NSP learns to predict optimal timestep selections from trajectory geometry
- Greedy oracle serves as upper bound on schedule efficiency
- Trajectory clustering suggests opportunity for conditional schedules

---

## Next Steps

After running the EDA:

1. **Review summary_report.txt** for key findings
2. **Examine phase-specific plots** for intuition
3. **Check clustering results** - do classes differ?
4. **Implement NSP network**:
   - Input: trajectory features (curvature, alignment, velocity)
   - Output: timestep selection or leap size prediction
   - Supervision: greedy oracle schedules
5. **Validate NSP** against oracle on test set

---

## Author Notes

- This EDA is **not meant to prove NSP optimal**—it demonstrates the geometry is real
- Heuristic observations, not formal proofs (as intended)
- MNIST is a simplified domain; expect richer patterns on high-res images
- All plots are saved for visual inspection and presentation

---

## License

Same as parent repository (flow-matching-mnist)

---

## Questions?

Refer to INIT.md for detailed phase specifications, or examine the source code in `src/` for implementation details.
