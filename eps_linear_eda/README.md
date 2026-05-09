# EPS Linear EDA - Theorem 3.1 Validation Suite

## Overview

This module implements an **Empirical Data Analysis (EDA) suite** to validate **Section III (Step Compression)** and **Theorem 3.1** from the "Adaptive and Neural Schedule Predictor" paper.

### What Does It Do?

For each time interval $[t_a, t_b]$, it compares:

$$\text{**Actual Error (LHS)**} = \|\hat{x}^N(t_b) - \hat{x}^1(t_b)\|$$

against the

$$\text{**Theoretical Bound (RHS)**} = \frac{\epsilon H^2}{2} \cdot \frac{e^{LH}-1}{LH} \cdot \left(1 + \frac{1}{N}\right)$$

where:
- $\epsilon = \sup_{t \in [t_a, t_b]} \|\ddot{x}^*(t)\|$ (curvature/acceleration bound)
- $L = \sup_t \|\nabla_x v_\theta(x, t)\|$ (Lipschitz constant of velocity field)
- $H = t_b - t_a$ (interval length)
- $N = 10$ (number of Euler sub-steps for comparison)

### Output: Triple Heatmap

The analysis produces **three visualization panels**:

1. **LHS Heatmap**: Actual measured error $\|\hat{x}^N - \hat{x}^1\|$ (X-axis: $t_b$, Y-axis: $t_a$)
2. **RHS Heatmap**: Theoretical bound computed from $\epsilon$ and $L$
3. **Tightness Ratio**: $\text{RHS} / \text{LHS}$ with **ALARM color** for violations

**Critical: Violations** (cells where LHS > RHS) indicate underestimated $L$ or $\epsilon$ for that region.

---

## Architecture

```
eps_linear_eda/
├── __init__.py
├── README.md                          [This file]
├── IMPLEMENTATION_SUMMARY.md          [Status & implementation details]
├── utils/
│   └── eps_linear_utils.py           [Core algorithms]
├── scripts/
│   └── compute_eps_linear_eda.py     [Production analysis]
├── tests/
│   └── test_eps_linear_pipeline.py   [Validation with dummy data]
└── results/
    ├── eps_linear_eda_analysis.png   [Triple heatmap visualization]
    ├── lhs_heatmap.npy               [Actual errors]
    ├── rhs_heatmap.npy               [Theoretical bounds]
    ├── tightness_ratio.npy           [RHS / LHS]
    ├── violation_mask.npy            [Boolean: LHS > RHS]
    ├── eps_grid.npy                  [Curvature values]
    ├── L_grid.npy                    [Lipschitz constants]
    └── statistics.json               [Summary metrics]
```

---

## How to Use

### Option 1: Quick Validation (Test Pipeline)

Run with dummy data to verify the implementation works (takes ~5-10 seconds):

```bash
python eps_linear_eda/tests/test_eps_linear_pipeline.py
```

Output: `eps_linear_eda/results/test_eps_linear_heatmaps.png`

**Test uses dummy velocity field** $v(x,t) = -2xt$ with analytical solution:
- Trajectory: $x(t) = x_0 e^{-t^2}$
- Curvature: $\|\ddot{x}(t)\| = |4x_0 t^2 - 2x_0| e^{-t^2}$
- Lipschitz: $L(t) = 2t$

### Option 2: Full Analysis (Production)

Run on saved Jacobian EDA trajectories (requires Jacobian EDA to be run first):

```bash
python eps_linear_eda/scripts/compute_eps_linear_eda.py
```

**Configuration (in script)**:
- Loads trajectories from: `jacobian_eda/results/trajectories.npy`
- Grid resolution: 50×50 time intervals
- Substeps: N=10 (compare 1-step vs 10-step Euler)
- Output: `eps_linear_eda/results/`

**Expected time**: ~2-5 minutes (depends on trajectory count)

---

## Mathematical Details

### Step 1: Curvature Computation

For each interval $[t_a, t_b]$, compute the **second-order finite difference**:

$$\epsilon(t_a, t_b) = \max_{t \in [t_a, t_b]} \left\| \frac{x(t + dt) - 2x(t) + x(t-dt)}{dt^2} \right\|$$

This estimates $\ddot{x}(t)$ = acceleration along the trajectory.

### Step 2: Lipschitz Constant

Use the **spectral norm** from Jacobian EDA:

$$L(t_a, t_b) = \text{mean}_{t \in [t_a, t_b]} \|\nabla_x v_\theta(x^*(t), t)\|$$

The Jacobian EDA already computes this via power iteration, so we reuse it.

### Step 3: Actual Error (LHS)

**N-step Euler** (accurate):
$$\hat{x}^N(t_b) = x_a + \frac{H}{N}v(...) + ... + \frac{H}{N}v(...) \quad (N \text{ substeps})$$

**1-step Euler** (coarse):
$$\hat{x}^1(t_b) = x_a + H \cdot v(x_a, t_a)$$

**Error**:
$$\text{LHS} = \|\hat{x}^N(t_b) - \hat{x}^1(t_b)\|$$

### Step 4: Theoretical Bound (RHS)

Apply Theorem 3.1 directly:

$$\text{RHS} = \frac{\epsilon H^2}{2} \cdot \frac{e^{LH} - 1}{LH} \cdot \left(1 + \frac{1}{N}\right)$$

with numerical safeguards:
- When $LH \to 0$: $(e^x - 1)/x \to 1$
- Overflow protection for large $LH$

### Step 5: Tightness Ratio

$$\text{Ratio} = \frac{\text{RHS}}{\text{LHS}}$$

**Interpretation**:
- $\text{Ratio} \approx 1$: Bound is **tight** (theory matches practice)
- $\text{Ratio} \gg 1$ (e.g., 100): Bound is **conservative** (proof may be pessimistic)
- $\text{Ratio} < 1$: **Violation** (indicates underestimated $L$ or $\epsilon$)

---

## Implementation Details

### Curvature via Finite Differences

```python
def compute_curvature_supremum(trajectory, t_start, t_end, time_grid):
    # Find time indices in [t_start, t_end]
    # For each trio (i-1, i, i+1):
    #   ẍ ≈ (x[i+1] - 2*x[i] + x[i-1]) / dt²
    # Return max ||ẍ||
```

### Lipschitz from Spectral Norm

```python
L = mean(spectral_norms[traj_idx, indices_in_interval])
```

The spectral norm is the largest eigenvalue of $\nabla_x v_\theta$, which bounds the Lipschitz constant.

### Error Computation

```python
# 1-step Euler
x_1step = x_start + H * v(x_start, t_start)

# N-step Euler
x_n = x_start
for i in range(N):
    dt = H / N
    v_curr = interpolate_velocity(trajectory, t_start + i*dt)
    x_n = x_n + dt * v_curr

# Error
error = ||x_n - x_1step||
```

### RHS Bound

```python
def compute_rhs_bound(epsilon, L, H, N=10):
    lh = L * H
    exp_factor = (exp(lh) - 1) / lh  # Safe version handles edge cases
    return (epsilon * H**2 / 2) * exp_factor * (1 + 1/N)
```

---

## Key Files

### `utils/eps_linear_utils.py` (~400 lines)

Core algorithms:

| Function | Purpose |
|----------|---------|
| `compute_curvature_supremum()` | Finite diff second derivative |
| `estimate_lipschitz_constant()` | Use spectral norm or FD |
| `euler_step()` | Single Euler step |
| `compute_actual_error()` | N-step vs 1-step error |
| `compute_rhs_bound()` | Theorem 3.1 bound |
| `compute_eps_linear_grid()` | Full grid computation |
| `identify_violations()` | Find LHS > RHS cells |

### `tests/test_eps_linear_pipeline.py` (~200 lines)

Dummy data validation:
- Generates 5 analytical trajectories
- Computes curvatures (max ||ẍ||)
- Validates error computation
- Creates test heatmaps
- Runs in ~5-10 seconds

### `scripts/compute_eps_linear_eda.py` (~300 lines)

Production pipeline:
- Loads Jacobian EDA artifacts
- Computes 50×50 grid of errors and bounds
- Creates triple heatmap visualization
- Saves all intermediate results
- Produces statistics JSON

---

## Artifact Outputs

After running, `eps_linear_eda/results/` contains:

| File | Type | Shape | Purpose |
|------|------|-------|---------|
| `eps_linear_eda_analysis.png` | PNG | 2x2 subplot | Triple heatmap + violations |
| `lhs_heatmap.npy` | NPY | (50, 50) | Actual measured errors |
| `rhs_heatmap.npy` | NPY | (50, 50) | Theoretical bounds |
| `tightness_ratio.npy` | NPY | (50, 50) | RHS / LHS ratios |
| `violation_mask.npy` | NPY | (50, 50) bool | Cells where LHS > RHS |
| `eps_grid.npy` | NPY | (50, 50) | Curvature values per cell |
| `L_grid.npy` | NPY | (50, 50) | Lipschitz constants per cell |
| `statistics.json` | JSON | metadata | Summary stats + pass/fail |

---

## Success Criteria

### Theorem 3.1 Validation

✓ **PASS**: No cells violate LHS > RHS (with numerical tolerance 1e-10)  
⚠️ **FAIL**: Any cell where LHS > RHS indicates theory does not match practice

### Tightness Assessment

- **Ratio ≈ 1-10**: Proof is **tight** → good practical bounds
- **Ratio ≈ 10-100**: Proof is **conservative** → maybe pessimistic
- **Ratio > 100**: Proof is **very conservative** → indicators may be too rough
- **Ratio < 1**: **Violation** → recheck $\epsilon$ and $L$ computation

---

## Interpreting Results

1. **Load the PNG visualization** (`eps_linear_eda_analysis.png`)
   - **Top-left (LHS)**: Where is the actual error large? (red regions = hard steps)
   - **Top-right (RHS)**: Where are the theoretical bounds large? (should align with LHS)
   - **Bottom-left (Ratio)**: Color intensity shows conservativeness
   - **Bottom-right (Violations)**: Any red X marks = bound violations

2. **Check statistics.json**
   ```json
   {
     "violations": 0,
     "violations_percentage": 0.0,
     "theorem_status": "PASS"
   }
   ```
   If `violations > 0`, investigate those cells in the numpy arrays.

3. **Extract specific cells** (for debugging)
   ```python
   import numpy as np
   lhs = np.load('eps_linear_eda/results/lhs_heatmap.npy')
   rhs = np.load('eps_linear_eda/results/rhs_heatmap.npy')
   eps = np.load('eps_linear_eda/results/eps_grid.npy')
   L = np.load('eps_linear_eda/results/L_grid.npy')
   
   # Find bad cell
   bad_cells = np.where(lhs > rhs + 1e-10)
   for i, j in zip(*bad_cells):
       print(f"Cell ({i},{j}): LHS={lhs[i,j]:.2e}, RHS={rhs[i,j]:.2e}, "
             f"ε={eps[i,j]:.2e}, L={L[i,j]:.2e}")
   ```

---

## Dependencies

- `numpy` - array operations
- `torch` - for potential Jacobian computation (fallback)
- `matplotlib` + `seaborn` - visualizations
- `tqdm` - progress bars

---

## Next Steps

1. **Run test pipeline first** (verify implementation)
   ```bash
   python eps_linear_eda/tests/test_eps_linear_pipeline.py
   ```

2. **Ensure Jacobian EDA is complete**
   ```bash
   ls jacobian_eda/results/trajectories.npy  # Must exist
   ```

3. **Run production analysis**
   ```bash
   python eps_linear_eda/scripts/compute_eps_linear_eda.py
   ```

4. **Inspect results**
   - View PNG visualization
   - Check JSON statistics
   - Load NPY files for detailed analysis

---

## References

- **Theorem 3.1** (Step Compression): "Adaptive and Neural Schedule Predictor" paper, Section III
- **Proof.md**: Full mathematical derivation of bounds
- **Jacobian EDA**: Uses pre-computed spectral norms for Lipschitz estimates

---

## Status: ✓ READY FOR TESTING

- [x] Utilities implemented (~400 lines)
- [x] Test pipeline created (~200 lines)
- [x] Production script ready (~300 lines)
- [x] Documentation complete
- [ ] Test pipeline executed
- [ ] Production analysis run

**Next**: Run test pipeline to validate
