"""
Jacobian computation utilities for Geodesic Deviation EDA.

Implements:
- Spectral norm estimation using finite differences
- Integration of spectral norm over time intervals
- Aggregate statistics across trajectories
"""

import torch
import numpy as np
from typing import Tuple, Callable


def finite_difference_spectral_norm(velocity_fn: Callable, x: torch.Tensor, t: torch.Tensor,
                                    num_samples: int = 10,
                                    eps: float = 1e-4,
                                    device: torch.device = torch.device('cpu')) -> float:
    """
    Estimate spectral norm ||∇_x v_θ(x,t)||_2 using finite differences.

    Algorithm:
    1. Generate random vectors u
    2. Estimate Jacobian-vector product: (v(x+eps*u) - v(x)) / eps ≈ J*u
    3. Return max ||J*u|| / ||u|| over random directions

    Args:
        velocity_fn: Velocity field function v_θ(x, t)
        x: State [batch, channels, height, width]
        t: Time value (scalar tensor)
        num_samples: Number of random directions to sample
        eps: Finite difference step size
        device: Torch device

    Returns:
        Estimated spectral norm (scalar)
    """
    batch_size = x.shape[0]
    x_shape = x.shape
    x_flat_size = int(np.prod(x_shape[1:]))  # Number of features per sample

    # Compute baseline velocity
    with torch.no_grad():
        v_base = velocity_fn(x, t)

    # Sample random directions and estimate norms
    max_norm = 0.0

    for _ in range(num_samples):
        # Random direction
        u = torch.randn(batch_size, *x.shape[1:], device=device)
        u_norm = torch.norm(u.reshape(batch_size, -1), p=2, dim=1, keepdim=True)
        u = u / (u_norm.reshape(batch_size, *([1]*len(x.shape[1:]))) + 1e-8)

        # Perturbed velocity
        x_pert = x + eps * u
        with torch.no_grad():
            v_pert = velocity_fn(x_pert, t)

        # Finite difference: (v(x+eps*u) - v(x)) / eps ≈ J*u
        jvp_fd = (v_pert - v_base) / eps

        # Norm of JVP
        jvp_norm = torch.norm(jvp_fd.reshape(batch_size, -1), p=2, dim=1).mean().item()
        max_norm = max(max_norm, jvp_norm)

    return max_norm


def integrate_spectral_norm(times: np.ndarray, spectral_norms: np.ndarray,
                             t_start: float, t_end: float) -> float:
    """
    Integrate spectral norm ||∇_x v_θ|| from t_start to t_end.

    Uses trapezoidal rule on precomputed spectral norms.

    Args:
        times: Array of time points [num_steps]
        spectral_norms: Array of spectral norms at each time [num_steps]
        t_start: Start of integration interval
        t_end: End of integration interval

    Returns:
        Integral value (scalar)
    """
    # Find indices in the interval [t_start, t_end]
    mask = (times >= t_start) & (times <= t_end)

    if mask.sum() < 2:
        return 0.0  # Not enough points

    t_sub = times[mask]
    s_sub = spectral_norms[mask]

    # Trapezoidal integration
    integral = np.trapz(s_sub, t_sub)
    return integral


def compute_geodesic_deviation_grid(trajectories: np.ndarray,
                                     spectral_norms: np.ndarray,
                                     grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute geodesic deviation integral for all (t_a, t_b) pairs.

    Args:
        trajectories: Precomputed trajectories [N, num_steps, ...]
        spectral_norms: Spectral norms at each step [N, num_steps]
        grid_size: Resolution of (t_a, t_b) grid (default 50)

    Returns:
        integrals: [N, grid_size, grid_size] - geodesic deviation for each (t_a, t_b)
        t_grid: [grid_size] - time grid points
        stats: dict with 'mean', 'std' heatmaps
    """
    N = trajectories.shape[0]
    num_steps = spectral_norms.shape[1]

    # Create time grid
    t_grid = np.linspace(0, 0.999, grid_size)  # Avoid t=1.0 numerical issues

    # Initialize integral tensor
    integrals = np.zeros((N, grid_size, grid_size))

    # Time points for integration
    times_full = np.linspace(0, 0.999, num_steps)

    # Compute integrals for all trajectories and all (t_a, t_b) pairs
    for traj_idx in range(N):
        for i, t_a in enumerate(t_grid):
            for j, t_b in enumerate(t_grid):
                if t_b >= t_a:  # Only upper triangle
                    integral = integrate_spectral_norm(times_full, spectral_norms[traj_idx], t_a, t_b)
                    integrals[traj_idx, i, j] = integral
                else:
                    integrals[traj_idx, i, j] = 0.0  # Lower triangle - set to 0

    # Compute statistics (only on upper triangle)
    mask = np.ones_like(integrals[0])
    for i in range(grid_size):
        for j in range(i):
            mask[i, j] = np.nan

    mean_integral = np.nanmean(integrals, axis=0)
    std_integral = np.nanstd(integrals, axis=0)

    # Handle NaN by replacing with 0
    mean_integral = np.nan_to_num(mean_integral, nan=0.0)
    std_integral = np.nan_to_num(std_integral, nan=0.0)

    stats = {
        'mean': mean_integral,
        'std': std_integral,
        'min': np.nanmin(integrals[~np.isnan(integrals)]),
        'max': np.nanmax(integrals[~np.isnan(integrals)])
    }

    return integrals, t_grid, stats


def compute_straightenability_mask(mean_integral: np.ndarray, std_integral: np.ndarray,
                                    epsilon: float = 0.1) -> np.ndarray:
    """
    Create a mask for "straightenable" regions where Mean + 1*StdDev < epsilon.

    Regions where this is true can use larger ODE steps (less "curvature").

    Args:
        mean_integral: Mean geodesic deviation heatmap
        std_integral: Std dev geodesic deviation heatmap
        epsilon: Threshold for straightenability

    Returns:
        Boolean mask [grid_size, grid_size]
    """
    threshold = mean_integral + std_integral
    mask = threshold < epsilon
    return mask
