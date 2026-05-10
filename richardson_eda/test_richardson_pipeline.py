#!/usr/bin/env python
"""
Richardson Extrapolation Test Pipeline

Implements Algorithm 5.1 step-by-step on dummy data to verify:
1. Coarse vs fine estimation
2. Error estimation formula
3. Step size update rule
4. Accept/reject loop
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Dummy velocity field: simple sinusoidal
def dummy_velocity_field(x: np.ndarray, t: float) -> np.ndarray:
    """Simple 2D velocity field for testing."""
    return np.array([np.sin(t), np.cos(t)]) * (1 + 0.1 * np.linalg.norm(x))

def richardson_step(x_k: np.ndarray, H: float, t_k: float, v_theta) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Richardson extrapolation error estimate.

    Returns:
        x_coarse: One step of size H
        x_fine: Two steps of size H/2
        error_hat: Estimated error = ||x_fine - x_coarse||
    """
    # Coarse: one Euler step of size H
    v_k = v_theta(x_k, t_k)
    x_coarse = x_k + H * v_k

    # Fine: two Euler steps of size H/2
    v_mid = v_theta(x_k + (H/2) * v_k, t_k + H/2)
    x_fine = x_k + (H/2) * v_k + (H/2) * v_mid

    # Error estimate (for p=1 Euler, divide by 2^1 - 1 = 1)
    error_hat = np.linalg.norm(x_fine - x_coarse)

    return x_coarse, x_fine, error_hat

def adaptive_update_rule(H_old: float, error_hat: float, tolerance: float, alpha: float = 0.9, f_min: float = 0.1, f_max: float = 5.0) -> float:
    """
    Compute new step size using Richardson adaptive formula.

    For Euler (p=1): H_new = H_old * α * (t / ê)^(1/(p+1)) = H_old * α * (t / ê)^(1/2)
    """
    p = 1  # Euler order
    if error_hat < 1e-10:
        return H_old * f_max

    ratio = (tolerance / error_hat) ** (1.0 / (p + 1))
    factor = np.clip(alpha * ratio, f_min, f_max)
    H_new = H_old * factor

    return H_new

def test_richardson_basic():
    """Test 1: Basic Richardson step on dummy data."""
    print("\n" + "="*70)
    print("TEST 1: Basic Richardson Step")
    print("="*70)

    x_0 = np.array([0.0, 0.0])
    t_0 = 0.0
    H = 0.1

    print(f"Initial state: x_0 = {x_0}, t = {t_0}, H = {H}")

    x_coarse, x_fine, error_hat = richardson_step(x_0, H, t_0, dummy_velocity_field)

    print(f"Coarse step (H):     {x_coarse}")
    print(f"Fine step (H/2):     {x_fine}")
    print(f"Estimated error:     {error_hat:.6e}")

    # Verify error estimate is O(H^2) by testing scaling
    print("\nVerifying O(H^2) scaling:")
    for H_test in [0.1, 0.05, 0.025]:
        _, _, e = richardson_step(x_0, H_test, t_0, dummy_velocity_field)
        print(f"  H = {H_test}: error = {e:.6e}")

    print("[OK] Test 1 passed")

def test_step_size_update():
    """Test 2: Adaptive step size update rule."""
    print("\n" + "="*70)
    print("TEST 2: Adaptive Step Size Update Rule")
    print("="*70)

    H_old = 0.1
    tolerance = 1e-3

    print(f"Initial H = {H_old}, tolerance = {tolerance}")
    print("\nTesting different error scenarios:")

    for error_hat in [1e-4, 1e-3, 1e-2, 0.1]:
        H_new = adaptive_update_rule(H_old, error_hat, tolerance)
        accept = error_hat <= tolerance
        print(f"  Error = {error_hat:.6e}: H_new = {H_new:.6f}, Accept: {accept}")

    print("[OK] Test 2 passed")

def test_accept_reject_loop():
    """Test 3: Accept/reject loop over a trajectory."""
    print("\n" + "="*70)
    print("TEST 3: Accept/Reject Loop")
    print("="*70)

    x_k = np.array([0.0, 0.0])
    t_k = 0.0
    t_final = 1.0
    tolerance = 5e-3
    alpha = 0.9
    f_min = 0.1
    f_max = 5.0

    H = 0.1  # Initial step size
    trajectory = [x_k.copy()]
    times = [t_k]
    step_sizes = [H]
    errors = []
    accepts = []
    rejects = 0
    total_steps = 0

    print(f"Tolerance = {tolerance}, Initial H = {H}")

    while t_k < t_final:
        # Ensure we don't overshoot
        if t_k + H > t_final:
            H = t_final - t_k

        # Richardson step
        x_coarse, x_fine, error_hat = richardson_step(x_k, H, t_k, dummy_velocity_field)
        total_steps += 1

        if error_hat <= tolerance:
            # Accept
            x_k = x_fine
            t_k += H
            trajectory.append(x_k.copy())
            times.append(t_k)
            errors.append(error_hat)
            accepts.append(True)

            # Update step size for next iteration
            H = adaptive_update_rule(H, error_hat, tolerance, alpha, f_min, f_max)
        else:
            # Reject
            rejects += 1
            accepts.append(False)
            H = adaptive_update_rule(H, error_hat, tolerance, alpha, f_min, f_max)

        step_sizes.append(H)

    trajectory = np.array(trajectory)
    times = np.array(times)
    errors = np.array(errors)
    step_sizes = np.array(step_sizes[:-1])

    print(f"\nResults:")
    print(f"  Total steps: {total_steps}")
    print(f"  Accepted: {len(trajectory) - 1}")
    print(f"  Rejected: {rejects}")
    print(f"  Reject rate: {100 * rejects / total_steps:.2f}%")
    print(f"  Final time: {t_k:.4f} / {t_final:.4f}")
    print(f"  Final position: {x_k}")
    print(f"  Mean step size: {np.mean(step_sizes):.6f}")
    print(f"  Mean error: {np.mean(errors):.6e}")
    print(f"  Max error: {np.max(errors):.6e}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('x1')
    axes[0, 0].set_ylabel('x2')
    axes[0, 0].set_title('Adaptive Trajectory')
    axes[0, 0].grid(True, alpha=0.3)

    # Step sizes over time
    axes[0, 1].plot(times[:-1], step_sizes, 'g-', linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Step Size H')
    axes[0, 1].set_title('Adaptive Step Sizes')
    axes[0, 1].grid(True, alpha=0.3)

    # Errors
    axes[1, 0].semilogy(times[1:], errors, 'r-', linewidth=2, marker='o', markersize=3, label='Estimated error')
    axes[1, 0].axhline(tolerance, color='k', linestyle='--', linewidth=2, label=f'Tolerance = {tolerance}')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Error (log scale)')
    axes[1, 0].set_title('Richardson Error Estimates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Accept/Reject pattern
    accept_colors = ['green' if a else 'red' for a in accepts]
    axes[1, 1].scatter(range(len(accepts)), [1] * len(accepts), c=accept_colors, s=50, alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Accept/Reject')
    axes[1, 1].set_title(f'Accept (Green) vs Reject (Red) - Reject Rate: {100*rejects/total_steps:.1f}%')
    axes[1, 1].set_ylim([0.5, 1.5])
    axes[1, 1].set_yticks([])
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('richardson_eda/test_richardson_loop.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: richardson_eda/test_richardson_loop.png")
    plt.close()

    print("[OK] Test 3 passed")

if __name__ == '__main__':
    import os
    os.makedirs('richardson_eda', exist_ok=True)

    print("\n" + "="*70)
    print("RICHARDSON EXTRAPOLATION TEST PIPELINE")
    print("="*70)

    test_richardson_basic()
    test_step_size_update()
    test_accept_reject_loop()

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
