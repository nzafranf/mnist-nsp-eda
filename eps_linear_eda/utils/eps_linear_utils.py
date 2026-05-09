"""
EPS Linear EDA Utilities

Implements curvature, Lipschitz, and error bound computations for Theorem 3.1 validation.

References:
- Theorem 3.1 (Step Compression with Error Bound)
- Section III (Step Compression) from "Adaptive and Neural Schedule Predictor" paper
"""

import numpy as np
import torch
from typing import Tuple, Optional


def compute_curvature_supremum(trajectory: np.ndarray, t_start: float, t_end: float,
                               time_grid: np.ndarray, sample_rate: int = 1) -> float:
    """
    Compute supremum (max) of curvature ||ẍ(t)|| = ||d²x/dt²|| over interval [t_start, t_end].

    Args:
        trajectory: [num_steps, dim] trajectory in flattened space
        t_start: start time (index or interpolated)
        t_end: end time (index or interpolated)
        time_grid: [num_steps] time values
        sample_rate: stride for sampling (default=1, use every point)

    Returns:
        supremum of curvature in interval
    """
    # Find indices that fall within [t_start, t_end]
    mask = (time_grid >= t_start) & (time_grid <= t_end)
    indices = np.where(mask)[0]

    if len(indices) < 3:
        return 0.0

    # Sample points
    indices = indices[::sample_rate]
    if len(indices) < 3:
        return 0.0

    # Compute second derivative via finite differences
    # ẍ ≈ (x[i+1] - 2*x[i] + x[i-1]) / dt²
    curvatures = []

    for i in range(1, len(indices) - 1):
        idx_prev = indices[i - 1]
        idx_curr = indices[i]
        idx_next = indices[i + 1]

        x_prev = trajectory[idx_prev]
        x_curr = trajectory[idx_curr]
        x_next = trajectory[idx_next]

        # Estimate dt from time grid
        dt = (time_grid[idx_next] - time_grid[idx_prev]) / 2.0

        if dt < 1e-8:
            continue

        # Second derivative
        x_double_dot = (x_next - 2 * x_curr + x_prev) / (dt ** 2)
        curvature = np.linalg.norm(x_double_dot)
        curvatures.append(curvature)

    if not curvatures:
        return 0.0

    return float(np.max(curvatures))


def estimate_lipschitz_constant(velocity_fn, x_trajectory: np.ndarray, t_val: float,
                               spectral_norm: float, delta: float = 1e-4) -> float:
    """
    Estimate local Lipschitz constant ||∇_x v_θ|| using spectral norm.

    In practice, the Lipschitz constant can be estimated as:
    1. Using spectral norm from Jacobian (preferred, already computed in Jacobian EDA)
    2. Via finite differences: ||v(x+δ) - v(x)|| / ||δ||

    Args:
        velocity_fn: velocity function v_θ (can be None if using spectral norm)
        x_trajectory: [dim] position
        t_val: time value
        spectral_norm: pre-computed spectral norm ||∇_x v_θ||
        delta: perturbation size for finite difference

    Returns:
        Estimated Lipschitz constant
    """
    # If spectral norm is provided and valid, use it directly
    # The Lipschitz constant of the velocity field is bounded by the spectral norm
    # of the Jacobian: ||v(x,t) - v(y,t)|| ≤ ||∇_x v|| * ||x - y||

    if spectral_norm > 0:
        return float(spectral_norm)

    # Fallback: use velocity_fn if provided
    if velocity_fn is None:
        return 0.5  # conservative default

    try:
        # Estimate via finite differences
        x_torch = torch.from_numpy(x_trajectory).float().unsqueeze(0)
        t_torch = torch.tensor([t_val], dtype=torch.float32)

        with torch.no_grad():
            v_x = velocity_fn(x_torch, t_torch)

            # Perturb
            noise = torch.randn_like(x_torch) * delta
            v_x_perturb = velocity_fn(x_torch + noise, t_torch)

            # Lipschitz estimate
            dv = torch.norm(v_x_perturb - v_x)
            dx = torch.norm(noise)

            if dx > 1e-10:
                return float((dv / dx).item())
    except Exception:
        pass

    return 0.5  # conservative default


def euler_step(x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """
    Single Euler step: x_new = x + v * dt

    Args:
        x: [dim] position
        v: [dim] velocity
        dt: time step

    Returns:
        [dim] new position
    """
    return x + v * dt


def interpolate_velocity_on_trajectory(trajectory: np.ndarray, time_grid: np.ndarray,
                                     t_eval: float) -> np.ndarray:
    """
    Interpolate velocity at time t_eval from trajectory points.
    Uses linear interpolation of positions to estimate velocity.

    Args:
        trajectory: [num_steps, dim]
        time_grid: [num_steps]
        t_eval: evaluation time

    Returns:
        [dim] interpolated velocity
    """
    # Find nearest times
    idx = np.searchsorted(time_grid, t_eval, side='left')

    if idx == 0:
        return np.zeros_like(trajectory[0])
    if idx >= len(time_grid):
        return np.zeros_like(trajectory[0])

    # Linear interpolation
    t_left = time_grid[idx - 1]
    t_right = time_grid[idx]

    if abs(t_right - t_left) < 1e-10:
        return np.zeros_like(trajectory[0])

    alpha = (t_eval - t_left) / (t_right - t_left)
    x_left = trajectory[idx - 1]
    x_right = trajectory[idx]

    # Velocity = (x_right - x_left) / (t_right - t_left)
    v = (x_right - x_left) / (t_right - t_left)
    return v


def compute_actual_error(trajectory: np.ndarray, time_grid: np.ndarray,
                        t_start_idx: int, t_end_idx: int,
                        n_substeps: int = 10) -> float:
    """
    Compute actual error ||x̂^N(t_b) - x̂^1(t_b)||

    N-step Euler vs 1-step Euler over interval.

    Args:
        trajectory: [num_steps, dim] reference trajectory
        time_grid: [num_steps] time values
        t_start_idx: start index
        t_end_idx: end index (must be > t_start_idx)
        n_substeps: number of substeps for N-step Euler

    Returns:
        Error norm
    """
    if t_end_idx <= t_start_idx:
        return 0.0

    # Starting position
    x_start = trajectory[t_start_idx]

    # Initial velocity at t_start
    v_start = interpolate_velocity_on_trajectory(trajectory, time_grid, time_grid[t_start_idx])

    # Total time interval
    H = time_grid[t_end_idx] - time_grid[t_start_idx]

    if H < 1e-10:
        return 0.0

    # 1-step Euler
    x_1step = euler_step(x_start.copy(), v_start, H)

    # N-step Euler
    x_n = x_start.copy()
    dt = H / n_substeps

    for step in range(n_substeps):
        t_curr = time_grid[t_start_idx] + step * dt
        v_curr = interpolate_velocity_on_trajectory(trajectory, time_grid, t_curr)
        x_n = euler_step(x_n, v_curr, dt)

    # Error
    error = np.linalg.norm(x_n - x_1step)
    return float(error)


def compute_rhs_bound(epsilon: float, L: float, H: float, N: int = 10) -> float:
    """
    Compute theoretical bound RHS from Theorem 3.1:

    RHS = (ε H² / 2) * (e^(LH) - 1) / (LH) * (1 + 1/N)

    Args:
        epsilon: curvature supremum ||ẍ||_max
        L: Lipschitz constant of velocity field
        H: interval length
        N: number of substeps

    Returns:
        RHS bound value
    """
    if epsilon < 1e-10 or H < 1e-10:
        return 0.0

    # Avoid numerical issues with small LH
    lh = L * H

    if abs(lh) < 1e-10:
        # When LH → 0: (e^x - 1)/x → 1
        exp_factor = 1.0
    else:
        try:
            exp_factor = (np.exp(lh) - 1) / lh
        except (OverflowError, RuntimeWarning):
            exp_factor = np.inf

    rhs = (epsilon * H ** 2 / 2) * exp_factor * (1 + 1 / N)
    return float(rhs)


def compute_eps_linear_grid(trajectories: np.ndarray, time_grid: np.ndarray,
                           spectral_norms: np.ndarray,
                           grid_size: int = 50,
                           n_substeps: int = 10,
                           velocity_fn=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute full triple heatmap grid for all time intervals [t_a, t_b].

    Args:
        trajectories: [N_traj, num_steps, dim]
        time_grid: [num_steps]
        spectral_norms: [N_traj, num_steps] Lipschitz constant estimates
        grid_size: resolution of t_a × t_b grid
        n_substeps: N for N-step Euler comparison
        velocity_fn: velocity function (optional, for fallback L estimation)

    Returns:
        lhs_heatmap: [grid_size, grid_size] actual errors
        rhs_heatmap: [grid_size, grid_size] theoretical bounds
        tightness_ratio: [grid_size, grid_size] RHS / LHS (with safe division)
        statistics: dict with metadata
    """
    N_traj = trajectories.shape[0]
    num_steps = trajectories.shape[1]

    # Create time index grid
    indices = np.linspace(0, num_steps - 1, grid_size, dtype=int)

    lhs_heatmap = np.zeros((grid_size, grid_size))
    rhs_heatmap = np.zeros((grid_size, grid_size))
    eps_grid = np.zeros((grid_size, grid_size))
    L_grid = np.zeros((grid_size, grid_size))

    # Aggregate across trajectories
    for i, idx_a in enumerate(indices):
        for j, idx_b in enumerate(indices):
            if idx_b <= idx_a:
                continue

            t_a = time_grid[idx_a]
            t_b = time_grid[idx_b]
            H = t_b - t_a

            # Aggregate over trajectories
            lhs_values = []
            rhs_values = []
            eps_values = []
            L_values = []

            for traj_idx in range(N_traj):
                # Curvature
                eps = compute_curvature_supremum(
                    trajectories[traj_idx], t_a, t_b, time_grid, sample_rate=1
                )
                eps_values.append(eps)

                # Lipschitz (use spectral norm)
                # Average spectral norm over the interval
                mask = (time_grid >= t_a) & (time_grid <= t_b)
                indices_interval = np.where(mask)[0]

                if len(indices_interval) > 0:
                    L = float(np.mean(spectral_norms[traj_idx, indices_interval]))
                else:
                    L = 0.5
                L_values.append(L)

                # Actual error (LHS)
                lhs = compute_actual_error(
                    trajectories[traj_idx], time_grid, idx_a, idx_b, n_substeps=n_substeps
                )
                lhs_values.append(lhs)

                # Theoretical bound (RHS)
                rhs = compute_rhs_bound(eps, L, H, N=n_substeps)
                rhs_values.append(rhs)

            # Store means
            lhs_heatmap[i, j] = float(np.mean(lhs_values)) if lhs_values else 0.0
            rhs_heatmap[i, j] = float(np.mean(rhs_values)) if rhs_values else 0.0
            eps_grid[i, j] = float(np.mean(eps_values)) if eps_values else 0.0
            L_grid[i, j] = float(np.mean(L_values)) if L_values else 0.0

    # Compute tightness ratio with safe division
    tightness_ratio = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if lhs_heatmap[i, j] > 1e-10:
                tightness_ratio[i, j] = rhs_heatmap[i, j] / lhs_heatmap[i, j]
            else:
                tightness_ratio[i, j] = 1.0  # Undefined but valid

    stats = {
        'lhs_min': float(np.min(lhs_heatmap[lhs_heatmap > 0])) if np.any(lhs_heatmap > 0) else 0.0,
        'lhs_max': float(np.max(lhs_heatmap)),
        'lhs_mean': float(np.mean(lhs_heatmap[lhs_heatmap > 0])) if np.any(lhs_heatmap > 0) else 0.0,
        'rhs_min': float(np.min(rhs_heatmap[rhs_heatmap > 0])) if np.any(rhs_heatmap > 0) else 0.0,
        'rhs_max': float(np.max(rhs_heatmap)),
        'rhs_mean': float(np.mean(rhs_heatmap[rhs_heatmap > 0])) if np.any(rhs_heatmap > 0) else 0.0,
        'violations': int(np.sum(lhs_heatmap > rhs_heatmap + 1e-10)),
        'num_trajectories': N_traj,
        'grid_size': grid_size,
        'num_steps': num_steps,
        'eps_grid': eps_grid,
        'L_grid': L_grid,
    }

    return lhs_heatmap, rhs_heatmap, tightness_ratio, stats


def identify_violations(lhs_heatmap: np.ndarray, rhs_heatmap: np.ndarray,
                       time_grid: np.ndarray, grid_size: int,
                       tolerance: float = 1e-10) -> np.ndarray:
    """
    Identify regions where LHS > RHS (bound violations).

    Args:
        lhs_heatmap: [grid_size, grid_size]
        rhs_heatmap: [grid_size, grid_size]
        time_grid: [num_steps]
        grid_size: grid resolution
        tolerance: numerical tolerance

    Returns:
        violation_mask: [grid_size, grid_size] boolean array
    """
    violation_mask = lhs_heatmap > (rhs_heatmap + tolerance)
    return violation_mask
