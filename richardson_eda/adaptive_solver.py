#!/usr/bin/env python
"""
Adaptive ODE Solver via Richardson Extrapolation (Algorithm 5.1)

Implements robust adaptive step-size control for Flow Matching trajectories.
Uses pre-computed trajectory artifacts to validate algorithm behavior.
"""

import numpy as np
from typing import Tuple, Callable, Dict
from dataclasses import dataclass

@dataclass
class SolverConfig:
    """Configuration for AdaptiveFlowSolver."""
    tolerance: float = 1e-3
    alpha: float = 0.9  # Safety factor
    f_min: float = 0.1  # Min step size multiplier
    f_max: float = 5.0  # Max step size multiplier
    max_steps: int = 10000
    verbose: bool = False

class AdaptiveFlowSolver:
    """
    Adaptive ODE solver using Richardson extrapolation.

    Algorithm 5.1 from PROOF.md:
    - Compute coarse (H) and fine (H/2) estimates
    - Estimate local error via Richardson extrapolation
    - Adapt step size: H_new = H_old * α * (t / ê)^(1/(p+1))
    - Accept if error <= tolerance, else reject and retry
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.p = 1  # Euler method order

    def richardson_step(
        self,
        x_k: np.ndarray,
        H: float,
        t_k: float,
        velocity_fn: Callable
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform one Richardson extrapolation step.

        Returns:
            x_coarse: One Euler step of size H
            x_fine: Two Euler steps of size H/2
            error_estimate: ||x_fine - x_coarse||
        """
        # Coarse: one step of size H
        v_k = velocity_fn(x_k, t_k)
        x_coarse = x_k + H * v_k

        # Fine: two steps of size H/2
        v_mid = velocity_fn(x_k + (H/2) * v_k, t_k + H/2)
        x_fine = x_k + (H/2) * v_k + (H/2) * v_mid

        # Error estimate (p=1 for Euler)
        error_estimate = np.linalg.norm(x_fine - x_coarse)

        return x_coarse, x_fine, error_estimate

    def compute_new_step_size(
        self,
        H_old: float,
        error_estimate: float
    ) -> float:
        """
        Compute adaptive step size using Richardson formula.

        H_new = H_old * α * (t / ê)^(1/(p+1))
        with clamping: f_min <= H_new/H_old <= f_max
        """
        tolerance = self.config.tolerance
        alpha = self.config.alpha
        f_min = self.config.f_min
        f_max = self.config.f_max

        if error_estimate < 1e-15:
            return H_old * f_max

        # For p=1 (Euler): exponent = 1/(p+1) = 1/2
        ratio = (tolerance / error_estimate) ** (1.0 / (self.p + 1))
        factor = np.clip(alpha * ratio, f_min, f_max)
        H_new = H_old * factor

        return H_new

    def solve(
        self,
        x0: np.ndarray,
        t_span: Tuple[float, float],
        velocity_fn: Callable,
        H0: float = 0.01
    ) -> Dict:
        """
        Solve ODE using adaptive Richardson extrapolation.

        Args:
            x0: Initial condition
            t_span: (t_start, t_end)
            velocity_fn: v_theta(x, t) function
            H0: Initial step size

        Returns:
            Dictionary with:
                - trajectory: Array of states [N, d]
                - times: Array of times [N]
                - step_sizes: Used step sizes
                - errors: Estimated errors at each accepted step
                - nfe: Number of function evaluations
                - n_accepted: Number of accepted steps
                - n_rejected: Number of rejected steps
                - reject_rate: Rejection rate
        """
        t_start, t_end = t_span
        x_k = x0.copy()
        t_k = t_start
        H = H0

        trajectory = [x_k.copy()]
        times = [t_k]
        errors = []
        step_sizes_used = []
        nfe = 0  # Function evaluations (2 per Richardson step)
        n_accepted = 0
        n_rejected = 0

        step_count = 0
        max_steps = self.config.max_steps

        while t_k < t_end and step_count < max_steps:
            # Don't overshoot final time
            if t_k + H > t_end:
                H = t_end - t_k

            # Richardson step (2 velocity evaluations)
            x_coarse, x_fine, error_est = self.richardson_step(x_k, H, t_k, velocity_fn)
            nfe += 3  # 1 coarse + 2 fine

            if error_est <= self.config.tolerance:
                # Accept step
                x_k = x_fine
                t_k += H
                trajectory.append(x_k.copy())
                times.append(t_k)
                errors.append(error_est)
                step_sizes_used.append(H)
                n_accepted += 1

                # Update step size for next iteration
                H = self.compute_new_step_size(H, error_est)
            else:
                # Reject step
                n_rejected += 1
                H = self.compute_new_step_size(H, error_est)

            step_count += 1

        reject_rate = n_rejected / (n_accepted + n_rejected) if (n_accepted + n_rejected) > 0 else 0

        return {
            'trajectory': np.array(trajectory),
            'times': np.array(times),
            'step_sizes': np.array(step_sizes_used),
            'errors': np.array(errors),
            'nfe': nfe,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'reject_rate': reject_rate,
            'config': self.config,
        }

def velocity_from_trajectory_finite_diff(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    traj_idx: int,
    smoothing_window: int = 3
) -> Callable:
    """
    Create velocity function from finite differences of trajectory with smoothing.

    Args:
        trajectories: [N_traj, N_steps, d] array
        t_grid: [N_steps] time grid
        traj_idx: Which trajectory to use
        smoothing_window: Window for smoothing (reduces noise)

    Returns:
        Callable v(x, t) that interpolates velocity at given (x, t)
    """
    traj = trajectories[traj_idx].astype(np.float64)  # [N_steps, d]

    # Compute velocity via finite differences
    dt = np.diff(t_grid)
    vel_raw = np.diff(traj, axis=0) / dt[:, None]

    # Smooth velocity with moving average
    velocities = np.zeros_like(traj)
    for i in range(len(traj)):
        if i == 0:
            velocities[i] = vel_raw[i]
        elif i == len(traj) - 1:
            velocities[i] = vel_raw[-1]
        else:
            start = max(0, i - smoothing_window // 2)
            end = min(len(vel_raw), i + smoothing_window // 2 + 1)
            velocities[i] = np.mean(vel_raw[start:end], axis=0)

    def velocity_fn(x: np.ndarray, t: float) -> np.ndarray:
        """Evaluate velocity at approximate time t via linear interpolation."""
        # Find closest time grid points for interpolation
        idx = np.searchsorted(t_grid, t, side='left')
        idx = np.clip(idx, 0, len(t_grid) - 1)

        if idx == 0:
            return velocities[0].copy()
        elif idx == len(t_grid):
            return velocities[-1].copy()
        else:
            # Linear interpolation between neighboring points
            t0, t1 = t_grid[idx - 1], t_grid[idx]
            v0, v1 = velocities[idx - 1], velocities[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-10)
            return (1 - alpha) * v0 + alpha * v1

    return velocity_fn

def reference_trajectory_fine_euler(
    x0: np.ndarray,
    t_span: Tuple[float, float],
    velocity_fn: Callable,
    n_steps: int = 10000
) -> Dict:
    """
    Generate reference trajectory using fine-grained Euler.
    Used as ground truth for comparing adaptive methods.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps

    trajectory = [x0.copy()]
    times = [t_start]
    x = x0.copy()

    for i in range(n_steps):
        t = t_start + i * dt
        v = velocity_fn(x, t)
        x = x + dt * v
        trajectory.append(x.copy())
        times.append(t + dt)

    return {
        'trajectory': np.array(trajectory),
        'times': np.array(times),
        'dt': dt,
        'n_steps': n_steps,
    }
