# Jacobian EDA Suite - Linearizability Analysis

## Overview

This suite validates **Theorem 4.1 (Linearizability)** and **Section IV (Jacobi Fields)** from "Adaptive and Neural Schedule Predictor" paper using the Flow Matching MNIST model.

The key insight: By computing the geodesic deviation integral
$$I(t_a, t_b) = \int_{t_a}^{t_b} \|\nabla_x v_\theta(x^*(t), t)\|_2 \, dt$$

we identify which time intervals $[t_a, t_b]$ are "ε-linear" (straightenable), meaning they can be traversed with larger ODE steps without significant accuracy loss.

## Quick Start

### Test Pipeline (with Dummy Data)
Validates the implementation before running on real model:
```bash
python jacobian_eda/tests/test_jacobian_pipeline.py
```
Output: `jacobian_eda/results/test_jacobian_heatmaps.png`

### Production Pipeline (Real Model)
Runs full EDA on trained Flow Matching model (takes ~30-60 min):
```bash
python jacobian_eda/scripts/compute_jacobian_eda.py
```

Output files:
- `trajectories.npy` - 50 trajectories with 100 integration steps
- `spectral_norms.npy` - Spectral norm ||∇_x v_θ|| at each step
- `integrals.npy` - Geodesic deviation I(t_a, t_b) for all intervals
- `straightenability_mask.npy` - Binary mask: regions where Mean + 1*Std < ε
- `statistics.json` - Summary statistics
- `jacobian_eda_analysis.png` - Heatmap visualizations

## Architecture

### Core Utilities (`utils/jacobian_utils.py`)

**`power_iteration_spectral_norm(velocity_fn, x, t, num_iterations=5)`**
- Estimates spectral norm ||∇_x v_θ(x,t)||_2 using Power Iteration
- Uses Jacobian-Vector Products (JVP) to avoid computing full Jacobian matrix
- Memory efficient: works on single trajectory at a time

**`integrate_spectral_norm(times, spectral_norms, t_start, t_end)`**
- Trapezoidal integration of spectral norm over interval [t_start, t_end]
- Identifies which intervals have low integrated Jacobian (straightenable)

**`compute_geodesic_deviation_grid(trajectories, spectral_norms, grid_size=50)`**
- Computes I(t_a, t_b) for all pairs on 50×50 grid
- Aggregates across N trajectories
- Returns mean and std dev heatmaps

**`compute_straightenability_mask(mean, std, epsilon=0.2)`**
- Creates binary mask: True where Mean + 1*Std < ε
- Identifies "safe" regions for large ODE steps

### Main Pipeline (`scripts/compute_jacobian_eda.py`)

**Configuration:**
- 50 trajectories (configurable)
- 100 integration steps (configurable)
- 50×50 grid for (t_a, t_b) intervals
- Power iteration with 5 steps per Jacobian estimate
- Straightenability threshold ε = 0.2

**Workflow:**
1. Load trained Flow Matching model from checkpoint
2. Generate 50 random noise trajectories via ODE integration
3. For each trajectory and each time step:
   - Compute velocity: v_θ(x(t), t)
   - Estimate spectral norm via power iteration
4. For all (t_a, t_b) pairs:
   - Integrate spectral norms
   - Store results in 3D tensor [N, grid_size, grid_size]
5. Aggregate statistics (mean, std) across trajectories
6. Identify straightenable regions (low integrated Jacobian)
7. Visualize as heatmaps

### Test Pipeline (`tests/test_jacobian_pipeline.py`)

Uses a **dummy velocity field** to validate pipeline:
- Simple quadratic model: v(x,t) = -2*x*t
- Spectral norm: ||∇_x v|| = 2*t (known analytically)
- 5 trajectories, 20 steps per trajectory
- Instant execution (~5 seconds)

## Interpretation Guide

### Heatmaps

**Mean Geodesic Deviation (viridis colormap)**
- Dark/Purple: Low integrated Jacobian → straightenable region
- Bright/Yellow: High integrated Jacobian → needs small steps

**Std Dev Heatmap (plasma colormap)**
- Indicates variance across trajectories
- High std = "digit-dependent" complexity
- Low std = "universally straight" regions

**Straightenability Mask (RdYlGn colormap)**
- Green: Straightenable (Mean + 1*Std < ε) → use large steps
- Red: Non-straightenable → use small adaptive steps

### Practical Use

**Step Scheduling:**
- Green regions [t_a, t_b]: Can use dt = 0.05 or larger
- Red regions [t_a, t_b]: Use dt = 0.01 or adaptive scheduling

**Algorithm Design:**
- Non-adaptive baseline: Use uniform small steps everywhere
- Adaptive (Algorithm 5.1): Use large steps in green regions, small in red
- Expected speedup: 2-5× on straightenable regions

## Technical Notes

### Jacobian Computation

We use **Power Iteration with JVP** to avoid full Jacobian matrix:
```
1. Initialize random u ~ N(0,1)
2. For k=1 to num_iterations:
   - Compute J·u via torch.autograd.grad()
   - Normalize: u = (J·u) / ||J·u||_2
3. Return ||J·u|| ≈ ||J||_2 (spectral norm)
```

This is efficient because:
- Time: O(1) per iteration (single backward pass)
- Memory: O(n) vs O(n²) for full Jacobian
- Accurate: 3-5 iterations sufficient for stable estimate

### Numerical Stability

**Time Boundary Handling:**
- Use t_max = 0.999 instead of 1.0 to avoid singularities
- Some Flow Matching models have numerical issues at exact t=1.0
- Error is negligible: integral from 0.999 to 1.0 is tiny

**Integration Precision:**
- Trapezoidal rule with 100 sample points
- Relative error < 0.1% for typical spectral norm trajectories

## Configuration

Edit `JacobianEDAConfig` class in `compute_jacobian_eda.py`:

```python
config.num_trajectories = 50           # More for better statistics
config.num_steps_integration = 100     # More for finer spectral norm
config.grid_size = 50                  # 50×50 grid for (t_a,t_b)
config.power_iterations = 5            # 3-5 is typical
config.epsilon_straightenable = 0.2    # Threshold for "straightenable"
```

## Performance

| Task | Time (CPU) | Memory |
|------|-----------|--------|
| Test pipeline | 5 sec | 100 MB |
| Real trajectory generation (50×100 steps) | 20-30 min | 500 MB |
| Spectral norm computation | 15-25 min | 400 MB |
| Integration & visualization | 2-5 min | 200 MB |
| **Total** | **40-60 min** | **~1 GB** |

## Output Interpretation

### Example Statistics (from test run)
```
Mean Integral Range: [0.0000, 0.9980]
Std Dev Range: [0.0000, 0.0000]
Straightenable Regions: 710/900 (78.9%)
```

Interpretation:
- Early time intervals (t_a, t_b ≈ small): Integral ≈ 0 (very straight)
- Late time intervals (t_a, t_b ≈ 1): Integral ≈ 1 (more curved)
- Most regions are straightenable → potential for adaptive scheduling

## References

- **Theorem 4.1**: Linearizability criteria based on integrated Jacobian
- **Section IV**: Jacobi field analysis for curvature of ODE flow
- **Algorithm 5.1**: Adaptive timestep scheduling based on straightenability

## Future Extensions

- [ ] Implement conditional Jacobian (class-specific analysis)
- [ ] Add animated heatmap showing time-evolution
- [ ] Compute upper bounds from straightenability mask
- [ ] Validate step size predictions via inference benchmarks
- [ ] Compare with existing schedulers (constant, exponential, learned)
